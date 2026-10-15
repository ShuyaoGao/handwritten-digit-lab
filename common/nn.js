// 浏览器端神经网络推理（纯 JavaScript，无依赖，file:// 可用）
// 模型用"层列表"描述（由 Python 导出，见 common/export.py），张量布局与 PyTorch 一致：[C,H,W] 或 [N]。
// 支持：conv（3×3 等，stride 1）、convT（转置卷积）、linear、relu、sigmoid、pool（2×2 最大池化）、
//       flatten、reshape、tap（记录中间结果；可按通道/神经元"敲掉"置零）。
var NN = (function () {
  "use strict";

  function f16(h) {
    var s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
  }

  // 解码 base64 的 float16 权重块；manifest: {名称: [偏移, 形状]}
  function loadWeights(manifest, b64, into) {
    var bin = atob(b64), n = bin.length >> 1, all = new Float32Array(n), W = into || {};
    for (var i = 0; i < n; i++) all[i] = f16(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
    Object.keys(manifest).forEach(function (k) {
      var off = manifest[k][0], shape = manifest[k][1], size = shape.reduce(function (a, b) { return a * b; }, 1);
      W[k] = { d: all.subarray(off, off + size), shape: shape };
    });
    return W;
  }

  function conv2d(x, s, w, b, pad) {
    var C = s[0], H = s[1], Wd = s[2], O = w.shape[0], k = w.shape[2];
    var OH = H + 2 * pad - k + 1, OW = Wd + 2 * pad - k + 1, y = new Float32Array(O * OH * OW), wd = w.d;
    for (var o = 0; o < O; o++) {
      var yo = o * OH * OW;
      y.fill(b.d[o], yo, yo + OH * OW);
      for (var c = 0; c < C; c++) {
        var xc = c * H * Wd;
        for (var ki = 0; ki < k; ki++) for (var kj = 0; kj < k; kj++) {
          var wv = wd[((o * C + c) * k + ki) * k + kj];
          if (wv === 0) continue;
          var i0 = Math.max(0, pad - ki), i1 = Math.min(OH, H + pad - ki), j0 = Math.max(0, pad - kj), j1 = Math.min(OW, Wd + pad - kj);
          for (var i = i0; i < i1; i++) {
            var xr = xc + (i + ki - pad) * Wd + kj - pad, yr = yo + i * OW;
            for (var j = j0; j < j1; j++) y[yr + j] += wv * x[xr + j];
          }
        }
      }
    }
    return { d: y, s: [O, OH, OW] };
  }

  // 转置卷积，w: [C,O,k,k]，输出边长 (H-1)*stride - 2*pad + k
  function convT(x, s, w, b, stride, pad) {
    var C = s[0], H = s[1], Wd = s[2], O = w.shape[1], k = w.shape[2];
    var OH = (H - 1) * stride - 2 * pad + k, OW = (Wd - 1) * stride - 2 * pad + k, y = new Float32Array(O * OH * OW), wd = w.d;
    for (var o = 0; o < O; o++) y.fill(b.d[o], o * OH * OW, (o + 1) * OH * OW);
    for (var c = 0; c < C; c++) for (var i = 0; i < H; i++) for (var j = 0; j < Wd; j++) {
      var xv = x[(c * H + i) * Wd + j];
      if (xv === 0) continue;
      for (var o2 = 0; o2 < O; o2++) for (var ki = 0; ki < k; ki++) {
        var oi = i * stride - pad + ki;
        if (oi < 0 || oi >= OH) continue;
        for (var kj = 0; kj < k; kj++) {
          var oj = j * stride - pad + kj;
          if (oj < 0 || oj >= OW) continue;
          y[(o2 * OH + oi) * OW + oj] += xv * wd[((c * O + o2) * k + ki) * k + kj];
        }
      }
    }
    return { d: y, s: [O, OH, OW] };
  }

  function pool2(x, s) {
    var C = s[0], H = s[1], Wd = s[2], OH = H >> 1, OW = Wd >> 1, y = new Float32Array(C * OH * OW);
    for (var c = 0; c < C; c++) for (var i = 0; i < OH; i++) for (var j = 0; j < OW; j++) {
      var p = c * H * Wd + 2 * i * Wd + 2 * j;
      y[(c * OH + i) * OW + j] = Math.max(x[p], x[p + 1], x[p + Wd], x[p + Wd + 1]);
    }
    return { d: y, s: [C, OH, OW] };
  }

  function linear(x, w, b) {
    var O = w.shape[0], I = w.shape[1], y = new Float32Array(O), wd = w.d;
    for (var o = 0; o < O; o++) {
      var acc = b.d[o], r = o * I;
      for (var i = 0; i < I; i++) acc += wd[r + i] * x[i];
      y[o] = acc;
    }
    return { d: y, s: [O] };
  }

  // 运行一个模型。prefix：权重名前缀（如 "cnn_m"）；layers：层列表；x：Float32Array（输入）；
  // opts.inShape：输入形状（默认 [1,28,28]）；opts.masks：{tap 名: [被敲掉的通道/神经元下标, ...]}
  // 返回 { out: Float32Array, taps: {名: {d, s}} }
  function run(W, prefix, layers, x, opts) {
    opts = opts || {};
    var t = { d: Float32Array.from(x), s: (opts.inShape || [1, 28, 28]).slice() }, taps = {};
    var P = function (n) {
      var v = W[prefix + "." + n];
      if (!v) throw new Error("缺少权重 " + prefix + "." + n);
      return v;
    };
    for (var li = 0; li < layers.length; li++) {
      var L = layers[li];
      switch (L.op) {
        case "conv": t = conv2d(t.d, t.s, P(L.w), P(L.b), L.pad || 0); break;
        case "convT": t = convT(t.d, t.s, P(L.w), P(L.b), L.stride, L.pad); break;
        case "linear": t = linear(t.d, P(L.w), P(L.b)); break;
        case "relu": for (var i = 0; i < t.d.length; i++) if (t.d[i] < 0) t.d[i] = 0; break;
        case "sigmoid": for (var j = 0; j < t.d.length; j++) t.d[j] = 1 / (1 + Math.exp(-t.d[j])); break;
        case "pool": t = pool2(t.d, t.s); break;
        case "flatten": t = { d: t.d, s: [t.d.length] }; break;
        case "reshape": t = { d: t.d, s: L.shape.slice() }; break;
        case "tap":
          var m = opts.masks && opts.masks[L.name];
          if (m && m.length) {
            var per = t.s.length === 1 ? 1 : t.d.length / t.s[0];
            m.forEach(function (c) { t.d.fill(0, c * per, (c + 1) * per); });
          }
          taps[L.name] = { d: Float32Array.from(t.d), s: t.s.slice() };
          break;
        default: throw new Error("未知层 " + L.op);
      }
    }
    return { out: t.d, taps: taps };
  }

  function softmax(a) {
    var mx = -Infinity, i, out = new Array(a.length), s = 0;
    for (i = 0; i < a.length; i++) mx = Math.max(mx, a[i]);
    for (i = 0; i < a.length; i++) { out[i] = Math.exp(a[i] - mx); s += out[i]; }
    for (i = 0; i < a.length; i++) out[i] /= s;
    return out;
  }

  function argmax(a) {
    var k = 0;
    for (var i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
    return k;
  }

  // 参数量（由层列表和权重形状算出，不手写）
  function paramCount(W, prefix, layers) {
    var n = 0;
    layers.forEach(function (L) {
      [L.w, L.b].forEach(function (k) { if (k) n += W[prefix + "." + k].d.length; });
    });
    return n;
  }

  return { loadWeights: loadWeights, run: run, softmax: softmax, argmax: argmax, paramCount: paramCount };
})();
if (typeof globalThis !== "undefined") globalThis.NN = NN;
