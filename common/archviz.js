// 网络结构图：每一层画成"真实特征图叠成的块"（参考 bbycroft.net/llm 的思路，做成 2.5D 的简化版）。
// 块的边长 ∝ 特征图边长，叠的层数 ∝ 通道数；全连接层画成一列。悬停显示连接方式、神经元数、参数公式；
// 点击播放"这一层怎么算"的动画（卷积核滑动、2×2 最大池化、全连接求和、反卷积盖章），全部用真实权重和当前输入。
var ARCHVIZ = (function () {
  "use strict";
  var KIND = {
    input: { name: "输入层", tok: "--k-io" }, output: { name: "输出层", tok: "--k-io" }, conv: { name: "卷积", tok: "--k-conv" },
    pool: { name: "池化", tok: "--k-pool" }, fc: { name: "全连接", tok: "--k-fc" }, convT: { name: "反卷积", tok: "--k-convT" },
    reshape: { name: "变形", tok: "--k-pool" }, code: { name: "编码", tok: "--k-code" },
  };
  var fmt = function (n) { return Number(n).toLocaleString(); };
  var prod = function (s) { return s.reduce(function (a, b) { return a * b; }, 1); };

  // 由层列表构造"块"：每个卷积 / 全连接 / 反卷积（连同后面的激活函数）、池化、变形各成一块，插入记录用的 tap
  function build(W, prefix, layers, inShape, opts) {
    opts = opts || {};
    var aug = [], blocks = [], n = 0, flat = null;
    if (!opts.noInput) blocks.push({ kind: "input", tap: null, shape: inShape.slice(), label: opts.inputLabel || "输入层" });
    for (var i = 0; i < layers.length; i++) {
      var L = layers[i];
      aug.push(L);
      if (L.op === "flatten") { flat = true; continue; }
      if (L.op === "conv" || L.op === "linear" || L.op === "convT") {
        var act = null;
        if (layers[i + 1] && (layers[i + 1].op === "relu" || layers[i + 1].op === "sigmoid")) { act = layers[i + 1].op; aug.push(layers[++i]); }
        var tap = "__a" + (n++);
        aug.push({ op: "tap", name: tap });
        var w = W[prefix + "." + L.w], b = W[prefix + "." + L.b];
        blocks.push({ kind: L.op === "linear" ? "fc" : L.op, tap: tap, L: L, w: w, b: b, act: act, flat: L.op === "linear" && flat, params: w.d.length + b.d.length });
        flat = null;
      } else if (L.op === "pool" || L.op === "reshape") {
        var t2 = "__a" + (n++);
        aug.push({ op: "tap", name: t2 });
        blocks.push({ kind: L.op, tap: t2, L: L, params: 0 });
      }
    }
    // 形状（由实际权重推出）
    var rows = ARCH.shapes(W, prefix, aug, inShape), byTap = {};
    rows.forEach(function (r) { if (r.op === "tap") byTap[r.name] = r.shape; });
    var prev = inShape;
    blocks.forEach(function (bk) {
      if (bk.tap) bk.shape = byTap[bk.tap];
      bk.inShape = prev;
      prev = bk.shape;
    });
    if (opts.codeLast) { var last = blocks[blocks.length - 1]; if (last.kind === "fc") last.code = true; }
    if (opts.outputLast) { var lo = blocks[blocks.length - 1]; lo.isOutput = true; }
    return { W: W, prefix: prefix, layers: aug, inShape: inShape, blocks: blocks, title: opts.title || "" };
  }

  function run(net, x, masks) {
    var r = NN.run(net.W, net.prefix, net.layers, x, { inShape: net.inShape, masks: masks });
    return {
      out: r.out, taps: r.taps,
      acts: net.blocks.map(function (b) { return b.tap ? r.taps[b.tap] : { d: Float32Array.from(x), s: net.inShape.slice() }; }),
    };
  }

  // ---------- 文字说明 ----------
  function title(b) {
    if (b.kind === "input") return b.label;
    if (b.isOutput && b.kind === "fc") return "输出层";
    if (b.code) return "编码层（瓶颈）";
    if (b.kind === "conv") return "卷积 " + b.w.shape[2] + "×" + b.w.shape[3];
    if (b.kind === "convT") return "反卷积 " + b.w.shape[2] + "×" + b.w.shape[3];
    if (b.kind === "pool") return "最大池化 2×2";
    if (b.kind === "reshape") return "变形";
    return "全连接";
  }
  function shapeText(s) { return s.length === 3 ? s[0] + "@" + s[1] + "×" + s[2] : s[0] + " 个"; }
  function describe(b) {
    var s = b.shape, N = prod(s), I = b.inShape, Iflat = prod(I), h = [];
    if (b.kind === "input") {
      h.push(b.label + "：" + (s.length === 3 ? s[1] + "×" + s[2] + " = " + fmt(N) + " 个像素，每个是 0–1 的灰度" : N + " 个数"));
      return h;
    }
    if (b.kind === "conv") {
      var O = b.w.shape[0], C = b.w.shape[1], k = b.w.shape[2], pad = b.L.pad || 0;
      h.push("<b>卷积层</b>：" + O + " 个 " + k + "×" + k + " 卷积核。每个卷积核同时看上一层全部 " + C + " 个通道（" + k + "×" + k + "×" + C + " = " + k * k * C + " 个权重），在图上逐格滑动（步长 1，四周补 " + pad + " 圈 0），各产生一张 " + s[1] + "×" + s[2] + " 的特征图" + (b.act === "relu" ? "，再经过 ReLU（负数变成 0）" : "") + "。");
      h.push("连接方式：每个输出神经元只连上一层 " + k + "×" + k + "×" + C + " = " + k * k * C + " 个位置（局部连接），同一张特征图上的神经元共用同一个卷积核（权重共享）。");
      h.push("神经元 " + O + "×" + s[1] + "×" + s[2] + " = <b>" + fmt(N) + "</b> 个；参数 " + O + "×" + C + "×" + k + "×" + k + " + " + O + "（偏置）= <b>" + fmt(b.params) + "</b> 个。");
    } else if (b.kind === "pool") {
      h.push("<b>池化层</b>：2×2 最大池化。每个通道上每 2×2 的格子只留最大的那个数，边长减半（" + I[1] + "×" + I[2] + " → " + s[1] + "×" + s[2] + "），通道数不变（" + s[0] + " 个）。");
      h.push("神经元 " + s[0] + "×" + s[1] + "×" + s[2] + " = <b>" + fmt(N) + "</b> 个；没有参数。作用：让网络对小的平移不敏感，同时减少计算量。");
    } else if (b.kind === "reshape") {
      h.push("<b>变形</b>：把上一层的 " + fmt(Iflat) + " 个数重新排成 " + s[0] + " 张 " + s[1] + "×" + s[2] + " 的图，数值不变，没有参数。这是编码器里\"拉直\"的逆操作。");
    } else if (b.kind === "fc") {
      var role = b.isOutput ? "<b>输出层</b>" : b.code ? "<b>编码层（瓶颈）</b>" : "<b>全连接层</b>";
      h.push(role + "：" + N + " 个神经元，每个都和上一层全部 " + fmt(Iflat) + " 个数相连（共 " + fmt(Iflat) + "×" + N + " = " + fmt(Iflat * N) + " 条连接）" +
        (b.flat && I.length === 3 ? "；上一层的 " + shapeText(I) + " 先被拉直成 " + fmt(Iflat) + " 个数" : "") + "。");
      if (b.act === "relu") h.push("每个神经元 = Σ（输入 × 权重）+ 偏置，再经过 ReLU。");
      else if (b.isOutput && N === 10) h.push("10 个数经过 softmax 变成\"是 0–9\"的概率。");
      else if (b.code) h.push("整张图被压到只剩这 " + N + " 个数，没有激活函数，可正可负。");
      h.push("参数 " + fmt(Iflat) + "×" + N + " + " + N + " = <b>" + fmt(b.params) + "</b> 个。");
    } else if (b.kind === "convT") {
      var Ci = b.w.shape[0], Co = b.w.shape[1], kk = b.w.shape[2], st = b.L.stride;
      h.push("<b>反卷积层（转置卷积）</b>：和卷积相反，把图放大。上一层每个像素拿一个 " + kk + "×" + kk + " 的权重块\"盖章\"到输出上，步长 " + st + "，相邻的章会重叠相加；" + Ci + " 个输入通道 → " + Co + " 个输出通道，边长 " + I[1] + " → " + s[1] + "。" +
        (b.act === "sigmoid" ? "最后经过 sigmoid 压到 0–1，就是画出来的灰度图。" : b.act === "relu" ? "再经过 ReLU。" : ""));
      h.push("神经元 " + s[0] + "×" + s[1] + "×" + s[2] + " = <b>" + fmt(N) + "</b> 个；参数 " + Ci + "×" + Co + "×" + kk + "×" + kk + " + " + Co + " = <b>" + fmt(b.params) + "</b> 个。");
    }
    return h;
  }

  // ---------- 绘制 ----------
  function View(wrap, info, opts) {
    this.wrap = wrap; this.info = info; this.opts = opts || {};
    this.cv = document.createElement("canvas");
    this.cv.className = "arch-canvas";
    wrap.innerHTML = "";
    wrap.appendChild(this.cv);
    this.hover = -1; this.items = []; this.groups = []; this.branches = []; this.acts = [];
    var self = this;
    this.cv.addEventListener("mousemove", function (e) { var i = self.hit(e); if (i !== self.hover) { self.hover = i; self.draw(); self.showInfo(); } });
    this.cv.addEventListener("mouseleave", function () { self.hover = -1; self.draw(); self.showInfo(); });
    this.cv.addEventListener("click", function (e) {
      var i = self.hit(e);
      if (i < 0) return;
      self.hover = i; self.draw(); self.showInfo();
      var it = self.items[i];
      if (it.block.kind !== "input" && it.block.kind !== "reshape" && self.acts[i] && self.acts[i - 1]) animate(it.block, self.acts[i - 1], self.acts[i], it.net);
    });
  }
  // segs: [{net, label, from（从第几块开始画）}]；branches: [{at: 块序号, label, img}]
  View.prototype.set = function (key, segs, branches) {
    if (key !== this.key) { this.key = key; this.hover = -1; }
    this.items = []; this.groups = [];
    var self = this;
    segs.forEach(function (sg) {
      var start = self.items.length;
      sg.net.blocks.forEach(function (b, j) { if (j >= (sg.from || 0)) self.items.push({ block: b, net: sg.net }); });
      self.groups.push({ a: start, b: self.items.length - 1, label: sg.label });
    });
    this.branches = branches || [];
  };
  View.prototype.update = function (acts) { this.acts = acts; this.draw(); this.showInfo(); };

  function layout(items, hasBranch) {
    var x = 14, pos = [];
    items.forEach(function (it) {
      var b = it.block, s = b.shape, w, h, dep = 0, slices = 1;
      if (s.length === 3) {
        w = h = 16 + 70 * Math.pow(s[1] / 28, 0.85);
        slices = Math.min(s[0], 6);
        dep = (slices - 1) * 5;
      } else {
        w = 14; h = Math.max(18, Math.min(110, 14 + 96 * Math.sqrt(s[0] / 1568)));
      }
      var slot = Math.max(w + dep, 64);
      pos.push({ x: x + (slot - w - dep) / 2, w: w, h: h, dep: dep, slices: slices, slot: slot, x0: x });
      x += slot + 18;
    });
    return { pos: pos, width: x, height: hasBranch ? 290 : 228, mid: 92 };
  }

  View.prototype.colors = function () {
    var cs = getComputedStyle(this.wrap), g = function (k) { return Pad.hexRgb(cs.getPropertyValue(k)); };
    return { surface: g("--surface"), sunken: g("--sunken"), ink: g("--ink"), ink2: g("--ink-2"), muted: g("--muted"), line: g("--line"), strong: g("--line-strong"),
      act: g("--accent-deep"), hi: g("--accent"), pos: g("--pos"), neg: g("--neg"), mono: cs.getPropertyValue("--f-mono"), body: cs.getPropertyValue("--f-body"),
      kind: function (k) { return g(KIND[k].tok); } };
  }

  View.prototype.draw = function () {
    if (!this.items.length) return;
    var L = layout(this.items, this.branches.length), dpr = window.devicePixelRatio || 1, cv = this.cv, C = this.colors(), self = this;
    // 宽屏上放不下时整体缩小（最多缩到 70%）；窄屏（手机）保持原大小，在框里左右滑
    var avail = this.wrap.clientWidth || L.width, sc = L.width > avail && avail >= 600 ? Math.max(0.7, avail / L.width) : 1;
    var cw = Math.max(L.width, avail / sc);
    this.scale = sc;
    cv.style.width = Math.round(cw * sc) + "px"; cv.style.height = Math.round(L.height * sc) + "px";
    cv.width = Math.round(cw * sc * dpr); cv.height = Math.round(L.height * sc * dpr);
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr * sc, 0, 0, dpr * sc, 0, 0);
    ctx.clearRect(0, 0, cw, L.height);
    this.layout = L;
    var rgb = function (c, a) { return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + (a == null ? 1 : a) + ")"; };
    var top = function (p) { return L.mid - p.h / 2; };
    // 分组（编码器 / 解码器 …）
    ctx.font = "600 12px " + C.body; ctx.textBaseline = "top"; ctx.textAlign = "left";
    this.groups.forEach(function (g) {
      if (!g.label) return;
      var a = L.pos[g.a], b = L.pos[g.b];
      ctx.strokeStyle = rgb(C.strong); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a.x0 + 4, 14); ctx.lineTo(a.x0 + 4, 10); ctx.lineTo(b.x0 + b.slot - 4, 10); ctx.lineTo(b.x0 + b.slot - 4, 14); ctx.stroke();
      ctx.fillStyle = rgb(C.ink2); ctx.fillText(g.label, a.x0 + 8, 14);
    });
    // 层与层之间的连线（淡）；悬停的层画出它的"感受野"
    for (var i = 1; i < this.items.length; i++) {
      var p0 = L.pos[i - 1], p1 = L.pos[i];
      ctx.strokeStyle = rgb(C.line); ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(p0.x + p0.w + p0.dep, top(p0) - p0.dep * 0.8 + 2); ctx.lineTo(p1.x, top(p1));
      ctx.moveTo(p0.x + p0.w + p0.dep, top(p0) + p0.h - p0.dep * 0.8 - 2); ctx.lineTo(p1.x, top(p1) + p1.h);
      ctx.stroke();
    }
    if (this.hover > 0) this.drawField(ctx, L, C, rgb);
    // 块
    this.items.forEach(function (it, i) {
      var b = it.block, p = L.pos[i], a = self.acts[i], y = top(p), s = b.shape;
      var hot = i === self.hover;
      if (s.length === 3) {
        var order = a ? topChannels(a, p.slices) : null;
        for (var k = p.slices - 1; k >= 0; k--) {
          var ox = p.x + k * 5, oy = y - k * 4;
          if (a) drawMap(ctx, a, order[k], ox, oy, p.w, C, k > 0);
          else { ctx.fillStyle = rgb(C.sunken); ctx.fillRect(ox, oy, p.w, p.w); }
          ctx.strokeStyle = rgb(hot ? C.hi : C.strong); ctx.lineWidth = hot ? 2 : 1; ctx.strokeRect(ox + 0.5, oy + 0.5, p.w - 1, p.w - 1);
        }
        if (s[0] > 1) { ctx.font = "11px " + C.mono; ctx.fillStyle = rgb(C.muted); ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText("×" + s[0], p.x + p.w + p.dep - 6, y - p.dep * 0.8 - 2); }
      } else {
        drawColumn(ctx, a, p.x, y, p.w, p.h, C, rgb, s[0]);
        ctx.strokeStyle = rgb(hot ? C.hi : C.strong); ctx.lineWidth = hot ? 2 : 1; ctx.strokeRect(p.x + 0.5, y + 0.5, p.w - 1, p.h - 1);
      }
      // 层类型小色条 + 文字
      var kind = b.isOutput && b.kind === "fc" ? "output" : b.code ? "code" : b.kind;
      ctx.fillStyle = rgb(C.kind(kind)); ctx.fillRect(p.x0 + 6, 150, p.slot - 12, 3);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillStyle = rgb(C.ink); ctx.font = "600 11.5px " + C.body; ctx.fillText(title(b), p.x0 + p.slot / 2, 158);
      ctx.fillStyle = rgb(C.ink2); ctx.font = "11px " + C.mono; ctx.fillText(shapeText(s), p.x0 + p.slot / 2, 174);
      ctx.fillStyle = rgb(C.muted); ctx.fillText(fmt(prod(s)), p.x0 + p.slot / 2, 189);
      if (b.params) { ctx.fillText(fmtK(b.params) + " 参数", p.x0 + p.slot / 2, 204); }
    });
    // 分支（识别网络各层 → 反演解码器 → 还原图）
    this.branches.forEach(function (br) {
      var p = L.pos[br.at];
      if (!p) return;
      var cx = p.x0 + p.slot / 2;
      ctx.strokeStyle = rgb(C.kind("convT")); ctx.setLineDash([3, 3]); ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(cx, 218); ctx.lineTo(cx, 236); ctx.stroke(); ctx.setLineDash([]);
      if (br.img) { var tmp = document.createElement("canvas"); paintGray(tmp, br.img, 28, 28, C.ink, C.surface); ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, cx - 20, 238, 40, 40); ctx.strokeStyle = rgb(C.strong); ctx.strokeRect(cx - 19.5, 238.5, 39, 39); }
      ctx.fillStyle = rgb(C.muted); ctx.font = "10.5px " + C.body; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      if (br.label) ctx.fillText(br.label, cx + 24, 258);
    });
  };
  function fmtK(n) { return n >= 10000 ? (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k" : fmt(n); }
  function topChannels(a, k) {
    var C = a.s[0], per = a.d.length / C, m = [];
    for (var c = 0; c < C; c++) { var s = 0; for (var i = 0; i < per; i++) s += a.d[c * per + i]; m.push([s, c]); }
    m.sort(function (x, y) { return y[0] - x[0]; });
    return m.slice(0, k).map(function (x) { return x[1]; });
  }
  var mapCache = null;
  function drawMap(ctx, a, ch, x, y, w, C, faint) {
    var H = a.s[1], Wd = a.s[2], per = H * Wd, mx = 1e-6, mn = 0, img = new Float32Array(per);
    for (var i = 0; i < per; i++) { var v = a.d[ch * per + i]; img[i] = v; if (v > mx) mx = v; if (v < mn) mn = v; }
    for (var j = 0; j < per; j++) img[j] = (img[j] - mn) / (mx - mn);
    mapCache = mapCache || document.createElement("canvas");
    paintGray(mapCache, img, Wd, H, faint ? mix(C.act, C.surface, 0.35) : C.act, C.surface);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(mapCache, x, y, w, w);
  }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function drawColumn(ctx, a, x, y, w, h, C, rgb, n) {
    ctx.fillStyle = rgb(C.sunken); ctx.fillRect(x, y, w, h);
    if (!a) return;
    var bins = Math.min(n, Math.floor(h / 2)), mx = 1e-6, mn = 0, d = a.d;
    for (var i = 0; i < d.length; i++) { if (d[i] > mx) mx = d[i]; if (d[i] < mn) mn = d[i]; }
    var signed = mn < -1e-6;
    for (var b = 0; b < bins; b++) {
      var i0 = Math.floor(b * n / bins), i1 = Math.max(i0 + 1, Math.floor((b + 1) * n / bins)), s = 0;
      for (var k = i0; k < i1; k++) s += d[k];
      s /= (i1 - i0);
      var col = signed ? (s >= 0 ? mix(C.surface, C.pos, s / Math.max(mx, -mn)) : mix(C.surface, C.neg, -s / Math.max(mx, -mn))) : mix(C.surface, C.act, s / mx);
      ctx.fillStyle = rgb(col);
      ctx.fillRect(x + 1, y + b * h / bins, w - 2, Math.max(1, h / bins - (bins <= 32 ? 1 : 0)));
    }
  }
  // 悬停层的"感受野"：卷积 = 上一层一个 k×k 小窗 → 本层一个点；全连接 = 上一层全部 → 本层一个点
  View.prototype.drawField = function (ctx, L, C, rgb) {
    var i = this.hover, b = this.items[i].block, p0 = L.pos[i - 1], p1 = L.pos[i], s0 = this.items[i - 1].block.shape, top = function (p) { return L.mid - p.h / 2; };
    var tx = p1.x + p1.w * 0.4, ty = top(p1) + p1.h * 0.4;
    ctx.fillStyle = rgb(C.hi, 0.18); ctx.strokeStyle = rgb(C.hi, 0.9); ctx.lineWidth = 1.2;
    if ((b.kind === "conv" || b.kind === "pool" || b.kind === "convT") && s0.length === 3) {
      var k = b.kind === "pool" ? 2 : b.kind === "conv" ? b.w.shape[2] : 1, cell = p0.w / s0[1];
      var fx = p0.x + p0.w * 0.4 - cell * k / 2, fy = top(p0) + p0.h * 0.4 - cell * k / 2, fw = cell * k;
      if (b.kind === "convT") { fw = cell; }
      ctx.fillRect(fx, fy, fw, fw); ctx.strokeRect(fx, fy, fw, fw);
      var tw = b.kind === "convT" ? p1.w / b.shape[1] * b.w.shape[2] : p1.w / b.shape[1];
      ctx.strokeRect(tx - tw / 2, ty - tw / 2, tw, tw);
      ctx.beginPath();
      [[fx, fy, tx - tw / 2, ty - tw / 2], [fx + fw, fy, tx + tw / 2, ty - tw / 2], [fx, fy + fw, tx - tw / 2, ty + tw / 2], [fx + fw, fy + fw, tx + tw / 2, ty + tw / 2]].forEach(function (l) { ctx.moveTo(l[0], l[1]); ctx.lineTo(l[2], l[3]); });
      ctx.stroke();
    } else {
      ctx.beginPath();
      for (var t = 0; t <= 12; t++) { ctx.moveTo(p0.x + p0.w + p0.dep, top(p0) - p0.dep * 0.8 + (p0.h) * t / 12); ctx.lineTo(tx, ty); }
      ctx.strokeStyle = rgb(C.hi, 0.35); ctx.stroke();
    }
  };
  View.prototype.hit = function (e) {
    if (!this.layout) return -1;
    var r = this.cv.getBoundingClientRect(), x = (e.clientX - r.left) / (this.scale || 1), L = this.layout;
    for (var i = 0; i < L.pos.length; i++) if (x >= L.pos[i].x0 - 9 && x < L.pos[i].x0 + L.pos[i].slot + 9) return i;
    return -1;
  };
  View.prototype.showInfo = function () {
    if (!this.info) return;
    if (this.hover < 0) {
      var tot = 0, n = this.items.length;
      this.items.forEach(function (it) { tot += it.block.params || 0; });
      var touch = document.body.classList.contains("touch");
      this.info.innerHTML = "共 " + n + " 层（含输入）· 参数合计 <b>" + fmt(tot) + "</b> 个。<span class=\"muted\">" +
        (touch ? "结构图可以左右滑；点任意一层，看它的连接方式并播放\"这一层怎么算\"。" : "鼠标移到任意一层看它的连接方式；点一下播放\"这一层怎么算\"。") + "</span>";
      return;
    }
    this.info.innerHTML = describe(this.items[this.hover].block).map(function (t) { return "<p>" + t + "</p>"; }).join("") +
      (this.items[this.hover].block.kind !== "input" && this.items[this.hover].block.kind !== "reshape" ? "<p class=\"muted\">点一下这一层，看它一步步怎么算。</p>" : "");
  };

  // ---------- 动画："这一层怎么算" ----------
  var A = null;
  function ensureModal() {
    if (A) return A;
    var el = document.createElement("div");
    el.className = "overlay arch-modal"; el.hidden = true;
    el.innerHTML = "<div class=\"sheet\"><div class=\"sheet-body\"><h2 class=\"am-title\"></h2><p class=\"am-sub small\"></p><div class=\"am-stage\"><canvas class=\"am-canvas\"></canvas></div><p class=\"am-text num\"></p></div>" +
      "<div class=\"sheet-foot\"><button class=\"btn\" type=\"button\" data-a=\"ch\">换一个通道</button><button class=\"btn\" type=\"button\" data-a=\"step\">单步</button><button class=\"btn\" type=\"button\" data-a=\"play\">暂停</button><button class=\"btn primary\" type=\"button\" data-a=\"close\">关闭</button></div></div>";
    document.body.appendChild(el);
    A = { el: el, cv: el.querySelector(".am-canvas"), raf: 0, playing: true };
    el.querySelector("[data-a=close]").onclick = close;
    el.querySelector("[data-a=play]").onclick = function () { A.playing = !A.playing; this.textContent = A.playing ? "暂停" : "继续"; };
    el.querySelector("[data-a=step]").onclick = function () { A.playing = false; el.querySelector("[data-a=play]").textContent = "继续"; if (A.step) A.step(1); };
    el.querySelector("[data-a=ch]").onclick = function () { if (A.next) A.next(); };
    el.addEventListener("click", function (e) { if (e.target === el) close(); });
    document.addEventListener("keydown", function (e) { if (!el.hidden && e.key === "Escape") { close(); e.stopPropagation(); } }, true);
    return A;
  }
  function close() { if (!A) return; cancelAnimationFrame(A.raf); A.el.hidden = true; A.step = A.next = null; }
  function isOpen() { return !!A && !A.el.hidden; }

  function animate(b, prev, cur, net) {
    var m = ensureModal(), C = View.prototype.colors.call({ wrap: document.body }), cv = m.cv, ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    m.el.hidden = false; m.playing = true; m.el.querySelector("[data-a=play]").textContent = "暂停";
    m.el.querySelector(".am-title").textContent = title(b) + " · " + shapeText(b.inShape) + " → " + shapeText(b.shape);
    var W0 = Math.min(720, m.el.querySelector(".am-stage").clientWidth || 640), H0 = Math.round(W0 * 0.5);
    cv.style.width = W0 + "px"; cv.style.height = H0 + "px"; cv.width = W0 * dpr; cv.height = H0 * dpr;
    var rgb = function (c, a) { return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + (a == null ? 1 : a) + ")"; };
    var text = m.el.querySelector(".am-text"), sub = m.el.querySelector(".am-sub");
    var chans = cur.s.length === 3 ? topChannels(cur, Math.min(cur.s[0], 8)) : [0], ci = 0, pos = 0, total = 1, out = null, ch = chans[0];
    var panel = function (x, y, size, arr, H, Wd, hiCells, lab, signed) {
      var cell = size / Math.max(H, Wd), mx = 1e-6, mn = 0;
      for (var i = 0; i < arr.length; i++) { if (arr[i] > mx) mx = arr[i]; if (arr[i] < mn) mn = arr[i]; }
      for (var r = 0; r < H; r++) for (var c = 0; c < Wd; c++) {
        var v = arr[r * Wd + c];
        if (v == null || isNaN(v)) { ctx.fillStyle = rgb(C.sunken); }
        else if (signed && mn < 0) ctx.fillStyle = rgb(v >= 0 ? mix(C.surface, C.pos, v / Math.max(mx, -mn)) : mix(C.surface, C.neg, -v / Math.max(mx, -mn)));
        else ctx.fillStyle = rgb(mix(C.surface, C.act, Math.max(0, (v - mn) / (mx - mn))));
        ctx.fillRect(x + c * cell, y + r * cell, cell + 0.5, cell + 0.5);
      }
      ctx.strokeStyle = rgb(C.strong); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, Wd * cell, H * cell);
      (hiCells || []).forEach(function (h) { ctx.strokeStyle = rgb(h[4] || C.hi); ctx.lineWidth = 2; ctx.strokeRect(x + h[1] * cell, y + h[0] * cell, h[3] * cell, h[2] * cell); });
      ctx.fillStyle = rgb(C.ink2); ctx.font = "12px " + C.body; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(lab, x, y + H * cell + 6);
      return cell;
    };
    var chanSum = function (a) { var Cn = a.s[0], per = a.d.length / Cn, s = new Float32Array(per); for (var c = 0; c < Cn; c++) for (var i = 0; i < per; i++) s[i] += a.d[c * per + i]; return s; };
    var setup, frame;
    if (b.kind === "conv") {
      var w = b.w, O = w.shape[0], Cin = w.shape[1], k = w.shape[2], pad = b.L.pad || 0, H = b.shape[1], Wd = b.shape[2], Hi = prev.s[1], Wi = prev.s[2];
      var inMap = Cin === 1 ? prev.d : chanSum(prev);
      setup = function () { out = new Float32Array(H * Wd).fill(NaN); pos = 0; total = H * Wd; sub.textContent = "卷积核 #" + ch + "（共 " + O + " 个）在 " + (Cin === 1 ? "输入图" : Cin + " 张输入图（左图显示它们的总和）") + " 上逐格滑动：每到一处，把 " + k + "×" + k + "×" + Cin + " 个数与卷积核对应相乘再相加，加上偏置，过 ReLU，就是右边特征图上的一个点。"; };
      frame = function () {
        var r = Math.floor(pos / Wd), c = pos % Wd, s = b.b.d[ch];
        for (var ic = 0; ic < Cin; ic++) for (var ki = 0; ki < k; ki++) for (var kj = 0; kj < k; kj++) {
          var ii = r + ki - pad, jj = c + kj - pad;
          if (ii >= 0 && ii < Hi && jj >= 0 && jj < Wi) s += w.d[((ch * Cin + ic) * k + ki) * k + kj] * prev.d[(ic * Hi + ii) * Wi + jj];
        }
        out[pos] = Math.max(0, s);
        draw3(function (x0, y0, sz) { return panel(x0, y0, sz, inMap, Hi, Wi, [[r - pad, c - pad, k, k]], Cin === 1 ? "输入 28×28" : "上一层 " + Cin + " 个通道（总和）"); },
          function (x0, y0, sz) { var kw = new Float32Array(k * k); for (var q = 0; q < k * k; q++) for (var ic2 = 0; ic2 < Cin; ic2++) kw[q] += w.d[(ch * Cin + ic2) * k * k + q]; return panel(x0, y0, sz, kw, k, k, null, "卷积核 " + k + "×" + k + (Cin > 1 ? "（" + Cin + " 层之和）" : ""), true); },
          function (x0, y0, sz) { return panel(x0, y0, sz, out, H, Wd, [[r, c, 1, 1]], "特征图 #" + ch + "  " + H + "×" + Wd); });
        text.textContent = "位置 (" + r + ", " + c + ")：Σ(窗口 × 卷积核) + 偏置 = " + s.toFixed(3) + " → ReLU → " + Math.max(0, s).toFixed(3);
      };
    } else if (b.kind === "pool") {
      var Hp = prev.s[1], Wp = prev.s[2], Ho = b.shape[1], Wo = b.shape[2];
      setup = function () { out = new Float32Array(Ho * Wo).fill(NaN); pos = 0; total = Ho * Wo; sub.textContent = "通道 #" + ch + "：每 2×2 的格子只留最大的数，边长减半。其余 " + (cur.s[0] - 1) + " 个通道各自做同样的事。"; };
      frame = function () {
        var r = Math.floor(pos / Wo), c = pos % Wo, base = ch * Hp * Wp, vals = [];
        for (var di = 0; di < 2; di++) for (var dj = 0; dj < 2; dj++) vals.push(prev.d[base + (2 * r + di) * Wp + 2 * c + dj]);
        out[pos] = Math.max.apply(null, vals);
        draw3(function (x0, y0, sz) { return panel(x0, y0, sz, prev.d.subarray(base, base + Hp * Wp), Hp, Wp, [[2 * r, 2 * c, 2, 2]], "上一层通道 #" + ch + "  " + Hp + "×" + Wp); }, null,
          function (x0, y0, sz) { return panel(x0, y0, sz, out, Ho, Wo, [[r, c, 1, 1]], "池化后  " + Ho + "×" + Wo); });
        text.textContent = "max(" + vals.map(function (v) { return v.toFixed(2); }).join(", ") + ") = " + out[pos].toFixed(2);
      };
    } else if (b.kind === "fc") {
      var I = prev.d.length, N = cur.d.length, wd = b.w.d, side = Math.ceil(Math.sqrt(I)), sideO = Math.ceil(Math.sqrt(N));
      var padArr = function (arr, n2) { var q = new Float32Array(n2).fill(NaN); q.set(arr); return q; };
      setup = function () { out = new Float32Array(N).fill(NaN); pos = 0; total = N; sub.textContent = "每个神经元都把上一层全部 " + fmt(I) + " 个数各乘一个自己的权重，再相加、加偏置" + (b.act === "relu" ? "、过 ReLU" : "") + "。左图橙色 = 对这个神经元贡献为正，蓝色 = 为负。"; };
      frame = function () {
        var j = pos, contrib = new Float32Array(I), s = b.b.d[j];
        for (var i = 0; i < I; i++) { contrib[i] = wd[j * I + i] * prev.d[i]; s += contrib[i]; }
        var v = b.act === "relu" ? Math.max(0, s) : b.act === "sigmoid" ? 1 / (1 + Math.exp(-s)) : s;
        out[j] = v;
        draw3(function (x0, y0, sz) { return panel(x0, y0, sz, padArr(contrib, side * side), side, side, null, "输入 × 权重（" + fmt(I) + " 个）", true); }, null,
          function (x0, y0, sz) { return panel(x0, y0, sz, padArr(out, sideO * sideO), sideO, sideO, [[Math.floor(j / sideO), j % sideO, 1, 1]], N + " 个神经元", b.act !== "relu"); });
        text.textContent = "神经元 #" + j + " = Σ(" + fmt(I) + " 个输入 × 权重) + 偏置 = " + s.toFixed(3) + (b.act ? " → " + (b.act === "relu" ? "ReLU" : "sigmoid") + " → " + v.toFixed(3) : "");
      };
    } else if (b.kind === "convT") {
      var wt = b.w, Ci = wt.shape[0], Co = wt.shape[1], kk = wt.shape[2], stv = b.L.stride, pd = b.L.pad, Hin = prev.s[1], Win = prev.s[2], Ho2 = b.shape[1], Wo2 = b.shape[2];
      var inSum = chanSum(prev);
      setup = function () { out = new Float32Array(Ho2 * Wo2).fill(b.b.d[ch]); pos = 0; total = Hin * Win; sub.textContent = "输出通道 #" + ch + "：上一层的每个像素（" + Ci + " 个通道一起）拿一个 " + kk + "×" + kk + " 的权重块，按步长 " + stv + " 盖到输出上，重叠的地方相加。所有像素盖完，再过 " + (b.act === "sigmoid" ? "sigmoid" : "ReLU") + "。"; };
      frame = function () {
        var r = Math.floor(pos / Win), c = pos % Win;
        for (var ic = 0; ic < Ci; ic++) {
          var xv = prev.d[(ic * Hin + r) * Win + c];
          if (!xv) continue;
          for (var ki = 0; ki < kk; ki++) for (var kj = 0; kj < kk; kj++) {
            var oi = r * stv - pd + ki, oj = c * stv - pd + kj;
            if (oi >= 0 && oi < Ho2 && oj >= 0 && oj < Wo2) out[oi * Wo2 + oj] += xv * wt.d[((ic * Co + ch) * kk + ki) * kk + kj];
          }
        }
        draw3(function (x0, y0, sz) { return panel(x0, y0, sz, inSum, Hin, Win, [[r, c, 1, 1]], "上一层 " + Ci + " 个通道（总和）" + Hin + "×" + Win); }, null,
          function (x0, y0, sz) { return panel(x0, y0, sz, out, Ho2, Wo2, [[r * stv - pd, c * stv - pd, kk, kk]], "输出通道 #" + ch + "（累加中）" + Ho2 + "×" + Wo2, true); });
        text.textContent = "输入像素 (" + r + ", " + c + ") → 盖到输出的 (" + (r * stv - pd) + "…" + (r * stv - pd + kk - 1) + ", " + (c * stv - pd) + "…" + (c * stv - pd + kk - 1) + ")";
      };
    }
    function draw3(f1, f2, f3) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W0, H0);
      var sz = Math.min(H0 - 30, (W0 - 60) * (f2 ? 0.4 : 0.48));
      f1(10, 6, sz);
      if (f2) f2(10 + sz + 24, 6 + sz * 0.3, sz * 0.32);
      f3(W0 - sz - 10, 6, sz);
      ctx.strokeStyle = rgb(C.muted); ctx.lineWidth = 1.2; ctx.beginPath();
      ctx.moveTo(10 + sz + 8, 6 + sz / 2); ctx.lineTo(W0 - sz - 22, 6 + sz / 2); ctx.lineTo(W0 - sz - 28, 6 + sz / 2 - 5); ctx.stroke();
    }
    m.step = function (n) { for (var i = 0; i < n && pos < total; i++) { frame(); pos++; } if (pos >= total) { pos = 0; setup(); } };
    m.next = function () { ci = (ci + 1) % chans.length; ch = chans[ci]; setup(); m.step(1); };
    var per = Math.max(1, Math.round(total / 240)), last = 0;
    setup(); per = Math.max(1, Math.round(total / 240)); m.step(1);
    cancelAnimationFrame(m.raf);
    (function loop(ts) {
      if (m.el.hidden) return;
      m.raf = requestAnimationFrame(loop);
      if (!m.playing || ts - last < 33) return;
      last = ts;
      m.step(per);
    })(0);
  }

  return { build: build, run: run, View: View, describe: describe, isOpen: isOpen, close: close };
})();
if (typeof globalThis !== "undefined") globalThis.ARCHVIZ = ARCHVIZ;
