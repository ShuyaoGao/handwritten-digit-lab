// 第二课 · 影子：2 维自编码器地图 → 解码器画影子；压缩多少；识别网络丢了什么。全部推理在浏览器里实时运行。
// 第一次进入第二课时才加载 assets/gen/core.js（file:// 下用 <script> 注入，不用 fetch）。元素 id 都带 m- 前缀。
(function () {
  "use strict";
  var impl = null, booting = false, queue = [];
  var MODE = { mdemo: "demo", map: "map", compress: "compress", invert: "invert" };
  function boot(cb) {
    if (impl) { cb(); return; }
    queue.push(cb);
    if (booting) return;
    booting = true;
    document.getElementById("m-brainBody").innerHTML = "<p class=\"loading\">正在加载第二课的模型……</p>";
    var s = document.createElement("script");
    s.src = "assets/gen/core.js";
    s.onload = function () { impl = start(); queue.splice(0).forEach(function (f) { f(); }); };
    s.onerror = function () { document.getElementById("m-brainBody").innerHTML = "<p class=\"loading\">assets/gen/core.js 加载失败，请确认 assets/gen/ 文件夹完整。</p>"; };
    document.body.appendChild(s);
  }
  UI.lesson({
    id: "shadow",
    name: "第二课 · 影子",
    short: "影子",
    tabs: [
      { id: "mdemo", name: "演示", title: "点在地图上自动游走" },
      { id: "map", name: "数字地图", short: "地图", op: true },
      { id: "compress", name: "压缩多少", short: "压缩" },
      { id: "invert", name: "识别网络丢了什么", short: "丢了什么" },
    ],
    onTab: function (id) { boot(function () { impl.setMode(MODE[id]); }); },
    onLeave: function () { if (impl) impl.leave(); },
    onLayout: function () { if (impl) impl.onLayout(); },
    onBrain: function (open) { if (impl) impl.onBrain(open); },
    menuItems: [["地图：铺满影子 / 测试图分布", function () { if (impl) impl.toggleGrid(); }], ["地图：全景 / 放大中心区", function () { if (impl) impl.toggleView(); }]],
    onKey: function (e) { if (impl) impl.onKey(e); },
  });

  function start() {
  var CORE = window.DM_CORE, R = CORE.result, AE = CORE.ae, CLF = CORE.classifier;
  var W = {};
  NN.loadWeights(CORE.weights.manifest, CORE.weights.b64, W);
  CORE.weights = null;
  var $ = function (id) { return document.getElementById("m-" + id); };
  var DIMS = Object.keys(AE).map(Number).sort(function (a, b) { return a - b; });
  var loaded = { core: true, compress: false, invert: false }, INV = null;
  var pct = function (v, d) { return (v * 100).toFixed(d == null ? 0 : d) + "%"; };

  function u8imgs(b64) {
    var s = atob(b64), n = s.length / 784, out = [];
    for (var i = 0; i < n; i++) { var a = new Float32Array(784); for (var j = 0; j < 784; j++) a[j] = s.charCodeAt(i * 784 + j) / 255; out.push(a); }
    return out;
  }
  var GAL = { y: CORE.gallery.y, orig: u8imgs(CORE.gallery.orig) };

  // 按需加载数据包（file:// 下用 <script> 注入，不用 fetch）
  function need(names, cb) {
    var todo = names.filter(function (n) { return !loaded[n]; });
    if (!todo.length) { cb(); return; }
    var n = todo[0], s = document.createElement("script");
    s.src = "assets/gen/" + n + ".js";
    s.onload = function () {
      var obj = window["DM_" + n.toUpperCase()];
      NN.loadWeights(obj.weights.manifest, obj.weights.b64, W);
      obj.weights = null;
      if (n === "invert") INV = obj;
      loaded[n] = true;
      need(names, cb);
    };
    s.onerror = function () { $("brainBody").innerHTML = "<p class=\"loading\">数据包 " + n + ".js 加载失败，请确认 assets/gen/ 文件夹完整。</p>"; };
    document.body.appendChild(s);
  }

  // ---------------- 模型 ----------------
  var masks = {};
  var M = function (m) { return masks[m] || (masks[m] = {}); };
  function enc(d, x) { return NN.run(W, "ae_" + d, AE[d].encoder, x, { masks: M("ae_" + d) }); }
  function dec(d, z) { return NN.run(W, "ae_" + d, AE[d].decoder, z, { inShape: [d], masks: M("ae_" + d) }); }
  function classify(x, masked) {
    var r = NN.run(W, "classifier", CLF.layers, x, { masks: masked ? M("classifier") : null });
    r.prob = NN.softmax(r.out); r.pred = NN.argmax(r.prob);
    return r;
  }
  function judge(img) { var r = classify(img, false); return { pred: r.pred, conf: r.prob[r.pred] }; }
  function mse(a, b) { var s = 0; for (var i = 0; i < 784; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return s / 784; }
  function textAE(d) {
    return {
      enc: ARCH.text(W, "ae_" + d, AE[d].encoder, [1, 28, 28], "输入 784（28×28）", "编码 {n} 个数"),
      dec: ARCH.text(W, "ae_" + d, AE[d].decoder, [d], "编码 " + d + " 个数", null, "输出 784（28×28）"),
      params: NN.paramCount(W, "ae_" + d, AE[d].encoder) + NN.paramCount(W, "ae_" + d, AE[d].decoder),
    };
  }

  // ---------------- 颜色 ----------------
  var C = {};
  function readColors() {
    // 第二课的强调色（橙）定义在 .lesson-shadow 上，所以从本课的面板上取色
    var cs = getComputedStyle($("brain"));
    ["--surface", "--sunken", "--ink", "--ink-2", "--muted", "--line", "--line-strong", "--accent", "--accent-deep", "--clf", "--ae", "--bad", "--paper"].forEach(function (k) { C[k] = Pad.hexRgb(cs.getPropertyValue(k)); });
    C.cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (i) { return cssColor("--c" + i); });
    C.mono = getComputedStyle(document.body).getPropertyValue("--f-mono");
  }
  var rgb = function (c, a) { return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + (a == null ? 1 : a) + ")"; };
  var mix = function (a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; };
  var ink = function (cv, arr) { paintGray(cv, arr, 28, 28, C["--ink"], C["--surface"]); };

  // ---------------- 状态 ----------------
  var Z = CORE.latent.z, ZY = CORE.latent.y, NZ = ZY.length, CENTERS = CORE.latent.centers;
  var st = {
    ready: false, mode: "demo", z: CENTERS[7].slice(), zSource: "demo", shadow: null, decRun: null, judge: null,
    x: null, fromUser: false, star: null, encRun: null, selD: 10, knob: null,
    mapMode: "pts", view: "zoom", focus: null, grid: null, demoT: 0, demoRaf: 0, interp: null, lastFrame: 0,
    tasks: { digits: {}, odd: false, star: false, interp: false },
  };

  // ---------------- 2 维地图 ----------------
  var EXT = (function () {
    var q = function (arr, p) { var s = arr.slice().sort(function (a, b) { return a - b; }); return s[Math.floor(p * (s.length - 1))]; };
    var xs = [], ys = [];
    for (var i = 0; i < NZ; i++) { xs.push(Z[2 * i]); ys.push(Z[2 * i + 1]); }
    var box = function (lo, hi, k) {
      var x0 = q(xs, lo), x1 = q(xs, hi), y0 = q(ys, lo), y1 = q(ys, hi), px = (x1 - x0) * k, py = (y1 - y0) * k;
      return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
    };
    // 没有正则化的自编码器，编码尺度很不均匀（"1" 拉得很远）：默认放大中心区，可切换全景
    return { zoom: box(0.08, 0.88, 0.08), full: box(0.005, 0.995, 0.04) };
  })();
  var AX = 30;
  function mapSize() { var r = $("mapBox").getBoundingClientRect(); return { w: r.width, h: r.height, dpr: window.devicePixelRatio || 1, left: r.left, top: r.top }; }
  function ext() { return EXT[st.view]; }
  function toPx(zx, zy, S) { var e = ext(); return [AX + (zx - e.x0) / (e.x1 - e.x0) * (S.w - AX - 8), 8 + (e.y1 - zy) / (e.y1 - e.y0) * (S.h - AX - 8)]; }
  function toZ(px, py, S) { var e = ext(); return [e.x0 + (px - AX) / (S.w - AX - 8) * (e.x1 - e.x0), e.y1 - (py - 8) / (S.h - AX - 8) * (e.y1 - e.y0)]; }
  function niceTicks(a, b) {
    var step = Math.pow(10, Math.floor(Math.log10((b - a) / 4))), m = (b - a) / step, s = m > 20 ? step * 5 : m > 8 ? step * 2 : step, out = [];
    for (var v = Math.ceil(a / s) * s; v <= b; v += s) out.push(+v.toFixed(6));
    return out;
  }
  function drawMapBase() {
    var S = mapSize();
    if (!S.w || $("cardMap").offsetParent === null) return;
    [$("mapBase"), $("mapTop")].forEach(function (cv) { cv.width = Math.round(S.w * S.dpr); cv.height = Math.round(S.h * S.dpr); });
    var ctx = $("mapBase").getContext("2d"), e = ext();
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.fillStyle = rgb(C["--surface"]); ctx.fillRect(0, 0, S.w, S.h);
    ctx.font = "11px " + C.mono; ctx.fillStyle = rgb(C["--muted"]); ctx.strokeStyle = rgb(C["--line"]); ctx.lineWidth = 1;
    niceTicks(e.x0, e.x1).forEach(function (t) { var x = toPx(t, 0, S)[0]; ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, S.h - AX); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(t, x, S.h - AX + 13); });
    niceTicks(e.y0, e.y1).forEach(function (t) { var y = toPx(0, t, S)[1]; ctx.beginPath(); ctx.moveTo(AX, y); ctx.lineTo(S.w - 8, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(t, AX - 4, y + 4); });
    ctx.textAlign = "center"; ctx.fillText("z₁", (S.w + AX) / 2, S.h - 4);
    ctx.save(); ctx.translate(9, (S.h - AX) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("z₂", 0, 0); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(AX, 8, S.w - AX - 8, S.h - AX - 8); ctx.clip();
    if (st.mapMode === "grid") {
      var n = UI.isMobile() ? 12 : 16, cw = (S.w - AX - 8) / n, ch = (S.h - AX - 8) / n;
      if (!st.grid || st.grid.view !== st.view || st.grid.n !== n) {
        var imgs = [];
        for (var r = 0; r < n; r++) for (var c = 0; c < n; c++) {
          var zx = e.x0 + (c + 0.5) / n * (e.x1 - e.x0), zy = e.y1 - (r + 0.5) / n * (e.y1 - e.y0);
          imgs.push(NN.run(W, "ae_2", AE[2].decoder, new Float32Array([zx, zy]), { inShape: [2] }).out);
        }
        st.grid = { view: st.view, n: n, imgs: imgs };
      }
      var tmp = document.createElement("canvas");
      st.grid.imgs.forEach(function (img, k) { ink(tmp, img); ctx.imageSmoothingEnabled = true; ctx.drawImage(tmp, AX + (k % n) * cw, 8 + Math.floor(k / n) * ch, cw, ch); });
    } else {
      var rad = Math.max(1.1, S.w / 430);
      if (st.focus != null) {
        ctx.fillStyle = rgb(C["--line"]);
        for (var i = 0; i < NZ; i++) { if (ZY[i] === st.focus) continue; var p = toPx(Z[2 * i], Z[2 * i + 1], S); ctx.fillRect(p[0] - rad, p[1] - rad, 2 * rad, 2 * rad); }
      }
      for (var j = 0; j < NZ; j++) {
        if (st.focus != null && ZY[j] !== st.focus) continue;
        var q2 = toPx(Z[2 * j], Z[2 * j + 1], S);
        ctx.fillStyle = rgb(C.cls[ZY[j]], st.focus == null ? 0.55 : 0.85);
        ctx.beginPath(); ctx.arc(q2[0], q2[1], rad, 0, 7); ctx.fill();
      }
    }
    // 类别中心标签：数字本身作为第二编码，不只靠颜色
    ctx.font = "600 13px " + C.mono; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    CENTERS.forEach(function (cz, c) {
      if (st.focus != null && st.focus !== c) return;
      var p = toPx(cz[0], cz[1], S);
      if (p[0] < AX || p[0] > S.w - 8 || p[1] < 8 || p[1] > S.h - AX) return;
      ctx.beginPath(); ctx.arc(p[0], p[1], 11, 0, 7); ctx.fillStyle = rgb(C["--surface"], 0.92); ctx.fill();
      ctx.lineWidth = 2.5; ctx.strokeStyle = rgb(C.cls[c]); ctx.stroke();
      ctx.fillStyle = rgb(C["--ink"]); ctx.fillText(c, p[0], p[1] + 1);
    });
    ctx.restore();
    ctx.textBaseline = "alphabetic";
    drawMapTop();
  }
  function drawMapTop() {
    var S = mapSize(), cv = $("mapTop");
    if (!S.w || !cv.width) return;
    var ctx = cv.getContext("2d");
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0); ctx.clearRect(0, 0, S.w, S.h);
    ctx.save(); ctx.beginPath(); ctx.rect(AX, 8, S.w - AX - 8, S.h - AX - 8); ctx.clip();
    if (st.interp) {
      var a = toPx(st.interp.a[0], st.interp.a[1], S), b = toPx(st.interp.b[0], st.interp.b[1], S);
      ctx.strokeStyle = rgb(C["--accent"], 0.8); ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
    }
    if (st.star) {
      var s = toPx(st.star[0], st.star[1], S);
      ctx.font = "22px " + C.mono; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.lineWidth = 3; ctx.strokeStyle = rgb(C["--surface"]); ctx.strokeText("★", s[0], s[1]); ctx.fillStyle = rgb(C["--ink"]); ctx.fillText("★", s[0], s[1]);
    }
    var p = toPx(st.z[0], st.z[1], S);
    ctx.strokeStyle = rgb(C["--ink"], 0.45); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(p[0], 8); ctx.lineTo(p[0], S.h - AX); ctx.moveTo(AX, p[1]); ctx.lineTo(S.w - 8, p[1]); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.fillStyle = rgb(C["--accent"]); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = rgb(C["--surface"]); ctx.stroke();
    ctx.restore();
  }

  function setZ(z, source) {
    st.z = [z[0], z[1]];
    st.zSource = source;
    st.decRun = dec(2, new Float32Array(st.z));
    st.shadow = st.decRun.out;
    st.judge = judge(st.shadow);
    if (source === "user") {
      if (st.judge.conf >= 0.8) st.tasks.digits[st.judge.pred] = true;
      if (st.judge.conf < 0.5) st.tasks.odd = true;
    }
    renderShadow();
    drawMapTop();
    if (st.mode === "demo" || st.mode === "map") renderBrain();
  }
  function renderShadow() {
    ink($("shadow"), st.shadow);
    $("zLine").textContent = "z = (" + st.z[0].toFixed(2) + ", " + st.z[1].toFixed(2) + ")";
    $("shadowVerdict").innerHTML = "识别网络觉得它像 <b>" + st.judge.pred + "</b>（把握 " + pct(st.judge.conf) + "）";
    renderTasks();
  }

  // 地图上的指针：电脑上鼠标悬停即可；手机上按住拖动
  var mapDown = false;
  function mapPointer(e) {
    var S = mapSize(), z = toZ(e.clientX - S.left, e.clientY - S.top, S);
    setZ(z, "user");
  }
  $("mapBox").addEventListener("pointerdown", function (e) {
    e.preventDefault();
    if (st.mode === "demo") UI.setTab("map");
    stopInterp();
    mapDown = true;
    try { $("mapBox").setPointerCapture(e.pointerId); } catch (err) { /* 忽略 */ }
    mapPointer(e);
  });
  $("mapBox").addEventListener("pointermove", function (e) {
    if (st.mode !== "map" || st.interp) return;
    if (mapDown || e.pointerType === "mouse") mapPointer(e);
  });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (t) { $("mapBox").addEventListener(t, function () { mapDown = false; }); });
  window.addEventListener("blur", function () { mapDown = false; });

  // 演示：点沿各数字中心之间的平滑曲线游走（"1" 的编码很远，不在放大区里，演示路线不经过它）
  var ROUTE = [7, 9, 4, 9, 7, 2, 3, 5, 8, 3, 0, 6, 5, 2].map(function (c) { return CENTERS[c]; });
  function routeAt(t) {
    var n = ROUTE.length, i = Math.floor(t) % n, f = t - Math.floor(t);
    var p0 = ROUTE[(i - 1 + n) % n], p1 = ROUTE[i], p2 = ROUTE[(i + 1) % n], p3 = ROUTE[(i + 2) % n];
    var cr = function (a, b, c, d) { return 0.5 * (2 * b + (-a + c) * f + (2 * a - 5 * b + 4 * c - d) * f * f + (-a + 3 * b - 3 * c + d) * f * f * f); };
    return [cr(p0[0], p1[0], p2[0], p3[0]), cr(p0[1], p1[1], p2[1], p3[1])];
  }
  function demoLoop(ts) {
    if (st.mode !== "demo" || UI.stopped()) return;
    st.demoRaf = requestAnimationFrame(demoLoop);
    if (UI.helpOpen() || ts - st.lastFrame < 50) return;
    var dt = st.lastFrame ? Math.min(100, ts - st.lastFrame) : 16;
    st.lastFrame = ts;
    st.demoT += dt / 2400;
    setZ(routeAt(st.demoT), "demo");
  }

  // 渐变：两个数字中心之间走直线
  function stopInterp() { if (st.interp) { cancelAnimationFrame(st.interp.raf); st.interp = null; drawMapTop(); } }
  function playInterp() {
    stopInterp();
    var a = CENTERS[+$("selA").value], b = CENTERS[+$("selB").value], t0 = performance.now();
    st.interp = { a: a, b: b, raf: 0 };
    var step = function (ts) {
      if (!st.interp || UI.stopped()) return;
      var t = Math.min(1, (ts - t0) / 2400);
      setZ([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], "interp");
      if (t < 1) st.interp.raf = requestAnimationFrame(step);
      else { st.tasks.interp = true; renderFilm(a, b); var keep = st.interp; setTimeout(function () { if (st.interp === keep) { st.interp = null; drawMapTop(); } }, 1200); renderTasks(); }
    };
    st.interp.raf = requestAnimationFrame(step);
    renderFilm(a, b);
  }
  function renderFilm(a, b) {
    var cv = $("film"), n = 10, ctx = cv.getContext("2d"), tmp = document.createElement("canvas");
    cv.width = 28 * n; cv.height = 28;
    for (var i = 0; i < n; i++) {
      var t = i / (n - 1), img = dec(2, new Float32Array([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])).out;
      ink(tmp, img); ctx.drawImage(tmp, i * 28, 0);
    }
  }

  // ---------------- 画板（随模式移动到对应位置） ----------------
  var pad = new Pad($("pad"), {
    lineWidth: 22,
    onDraw: function () { onInput(pad.toInput(), true, false); },
    onEnd: function () { onInput(pad.toInput(), true, true); },
  });
  $("padBox").addEventListener("pointerdown", function () { $("padHint").hidden = true; if (st.mode === "demo") UI.setTab("map"); }, true);
  function placePad() {
    var slot = { demo: "slotMap", map: "slotMap", compress: "slotCompress", invert: "slotInvert" }[st.mode];
    $(slot).appendChild($("inputCard"));
  }
  function loadSample(i) {
    i = (i + GAL.orig.length) % GAL.orig.length;
    pad.setSample(GAL.orig[i]);
    $("padHint").hidden = true;
    onInput(GAL.orig[i], false, true);
  }
  function onInput(x, fromUser, final) {
    st.x = x;
    st.fromUser = fromUser;
    st.knob = null;
    if (st.mode === "map" || st.mode === "demo") {
      if (!x) { st.star = null; st.encRun = null; drawMapTop(); renderBrain(); return; }
      st.encRun = enc(2, x);
      st.star = [st.encRun.out[0], st.encRun.out[1]];
      if (final && fromUser) st.tasks.star = true;
      stopInterp();
      setZ(st.star, "star");
    } else if (st.mode === "compress") runCompress();
    else if (st.mode === "invert") runInvert();
  }
  $("btnClear").onclick = function () { pad.clear(); $("padHint").hidden = false; onInput(null, true, true); };
  $("btnUndo").onclick = function () { pad.undo(); onInput(pad.toInput(), true, true); };
  $("btnSample").onclick = function () { loadSample(Math.floor(Math.random() * GAL.orig.length)); };

  // ---------------- 压缩多少 ----------------
  var cmp = null;
  function ensureInput() { if (!st.x) loadSample(GAL.y.indexOf(7)); }
  function runCompress() {
    if (!loaded.compress) return;
    if (!st.x) { $("reconGrid").innerHTML = "<p class=\"small\">写一个数字或点\"示例\"。</p>"; return; }
    cmp = {};
    DIMS.forEach(function (d) {
      var e = enc(d, st.x), z = e.out, zz = d === st.selD && st.knob ? st.knob : z, r = dec(d, zz);
      cmp[d] = { enc: e, z: z, zz: zz, dec: r, out: r.out, mse: mse(r.out, st.x), judge: judge(r.out) };
    });
    renderCompress();
    renderBrain();
  }
  function renderCompress() {
    var orig = judge(st.x);
    var cards = [{ k: "orig", name: "原图", sub: "784 个数 · 认作 <em>" + orig.pred + "</em>" }].concat(DIMS.map(function (d) {
      var c = cmp[d];
      return { k: d, name: d + " 个数", sub: "误差 <em>" + c.mse.toFixed(4) + "</em><br>认作 <em class=\"" + (c.judge.pred !== orig.pred ? "miss" : "") + "\">" + c.judge.pred + "</em>" + (d === st.selD && st.knob ? " · <span class=\"tweak\">已拧旋钮</span>" : "") };
    }));
    $("reconGrid").innerHTML = cards.map(function (c) {
      return "<button type=\"button\" class=\"rc\" data-k=\"" + c.k + "\" aria-pressed=\"" + (c.k === st.selD) + "\"><canvas class=\"px\"></canvas><b class=\"" + (c.k === "orig" ? "" : "ae") + "\">" + c.name + "</b><span>" + c.sub + "</span></button>";
    }).join("");
    Array.prototype.forEach.call($("reconGrid").children, function (el) {
      var k = el.dataset.k;
      ink(el.querySelector("canvas"), k === "orig" ? st.x : cmp[k].out);
      if (k !== "orig") el.onclick = function () { st.selD = +k; st.knob = null; runCompress(); };
    });
    renderKnobs();
  }
  function renderKnobs() {
    var d = st.selD, c = cmp[d], s = AE[d].stats, n = Math.min(d, 8);
    $("knobTitle").textContent = d + " 维自编码器" + (d > n ? "（共 " + d + " 个数，这里给出前 " + n + " 个）" : "（共 " + d + " 个数）");
    if ($("knobs").dataset.d !== String(d) || $("knobs").children.length !== n) {
      $("knobs").dataset.d = d;
      $("knobs").innerHTML = "";
      for (var i = 0; i < n; i++) {
        var lab = document.createElement("label");
        lab.className = "knob"; lab.htmlFor = "m-knob" + i;
        lab.innerHTML = "<span>z" + (i + 1) + "</span><input type=\"range\" id=\"m-knob" + i + "\" step=\"any\"><b></b>";
        $("knobs").appendChild(lab);
        lab.querySelector("input").oninput = onKnob;
      }
    }
    Array.prototype.forEach.call($("knobs").children, function (lab, i) {
      var inp = lab.querySelector("input"), v = c.zz[i];
      inp.min = s.mean[i] - 3 * s.std[i]; inp.max = s.mean[i] + 3 * s.std[i]; inp.value = v;
      lab.querySelector("b").textContent = v.toFixed(2);
    });
  }
  function onKnob() {
    var d = st.selD, z = Float32Array.from(st.knob || cmp[d].z);
    Array.prototype.forEach.call($("knobs").children, function (lab, i) { z[i] = +lab.querySelector("input").value; });
    st.knob = z;
    var c = cmp[d];
    c.zz = z; c.dec = dec(d, z); c.out = c.dec.out; c.mse = mse(c.out, st.x); c.judge = judge(c.out);
    Array.prototype.forEach.call($("knobs").children, function (lab, i) { lab.querySelector("b").textContent = z[i].toFixed(2); });
    var el = $("reconGrid").querySelector("[data-k=\"" + d + "\"]");
    if (el) {
      ink(el.querySelector("canvas"), c.out);
      el.querySelector("span").innerHTML = "误差 <em>" + c.mse.toFixed(4) + "</em><br>认作 <em>" + c.judge.pred + "</em> · <span class=\"tweak\">已拧旋钮</span>";
    }
    renderBrain();
  }
  $("btnKnobReset").onclick = function () { st.knob = null; runCompress(); };

  // ---------------- 识别网络丢了什么 ----------------
  var inv = null;
  function runInvert() {
    if (!loaded.invert || !loaded.compress) return;
    if (!st.x) { $("invGrid").innerHTML = "<p class=\"small\">写一个数字或点\"示例\"。</p>"; return; }
    var c = classify(st.x, true), base = judge(st.x);
    var feats = { C1: c.taps.C1, C2: c.taps.C2, F1: c.taps.F1, logits: { d: c.out } };
    var items = ["C1", "C2", "F1", "logits"].map(function (k) {
      var out = NN.run(W, "inv_" + k, INV.layers[k], feats[k].d, { inShape: INV.in_shape[k] }).out;
      return { name: { C1: "从 C1 还原", C2: "从 C2 还原", F1: "从 F1 还原", logits: "从输出层还原" }[k], fam: "clf", n: INV.in_shape[k].reduce(function (a, b) { return a * b; }, 1), out: out };
    });
    [10, 2].forEach(function (d) {
      var out = NN.run(W, "ae_" + d, AE[d].decoder, NN.run(W, "ae_" + d, AE[d].encoder, st.x).out, { inShape: [d] }).out;
      items.push({ name: "自编码器 " + d + " 维", fam: "ae", n: d, out: out });
    });
    items.forEach(function (it) { it.judge = judge(it.out); it.mse = mse(it.out, st.x); });
    inv = { clf: c, items: items, base: base };
    $("invGrid").innerHTML = items.map(function (it) {
      return "<div class=\"rc\"><canvas class=\"px\"></canvas><b class=\"" + it.fam + "\">" + it.name + "</b><span>" + it.n + " 个数 · 误差 <em>" + it.mse.toFixed(4) + "</em><br>认作 <em class=\"" + (it.judge.pred !== base.pred ? "miss" : "") + "\">" + it.judge.pred + "</em>（" + pct(it.judge.conf) + "）</span></div>";
    }).join("");
    Array.prototype.forEach.call($("invGrid").children, function (el, i) { ink(el.querySelector("canvas"), items[i].out); });
    renderBrain();
  }

  // 条形图（测试集 1 万张，数字来自 result.json）
  function bars(el, rows, key, maxV, fmt, ref) {
    el.innerHTML = rows.map(function (m, i) {
      var v = m.r[key].value, w = Math.max(0.5, v / maxV * 100), sep = i > 0 && rows[i - 1].fam !== m.fam ? " sep" : "";
      return "<div class=\"brow" + sep + "\" title=\"" + m.name + "（" + m.dim + " 个数）：" + fmt(v) + "\"><span class=\"lab-n\">" + m.short + "<span class=\"small\"> · " + m.dim + "</span></span><span class=\"track\"><i class=\"" + m.fam + "\" style=\"width:" + w + "%\"></i></span><span class=\"val\">" + fmt(v) + "</span></div>";
    }).join("");
    if (ref == null) return;
    var line = document.createElement("div");
    line.className = "refline";
    el.appendChild(line);
    var place = function () {
      var tr = el.querySelectorAll(".track"), eb = el.getBoundingClientRect(), f = tr[0].getBoundingClientRect(), l = tr[tr.length - 1].getBoundingClientRect();
      line.style.left = (f.left - eb.left + f.width * ref / maxV) + "px";
      line.style.top = (f.top - eb.top - 3) + "px";
      line.style.height = (l.bottom - f.top + 6) + "px";
    };
    place();
    if (window.ResizeObserver) new ResizeObserver(place).observe(el);
  }
  var AE_ROWS = DIMS.slice().reverse().map(function (d) { return { fam: "ae", name: "自编码器 " + d + " 维", short: "AE " + d + " 维", dim: d, r: R.autoencoder[String(d)] }; });
  var INV_ROWS = ["C1", "C2", "F1", "logits"].map(function (k) { return { fam: "clf", name: "识别网络 " + k, short: k === "logits" ? "输出层" : k, dim: R.inversion[k].dim, r: R.inversion[k] }; }).concat(AE_ROWS);
  function panels(el, rows) {
    var mx = Math.max.apply(null, rows.map(function (m) { return m.r.mse.value; })) * 1.05;
    el.innerHTML = "<div class=\"panel\"><h3>像素误差 MSE <span class=\"small\">越短越像原图</span></h3><div class=\"bars\"></div></div>" +
      "<div class=\"panel\"><h3>还原图被识别网络认对 <span class=\"small\">越长越好</span></h3><div class=\"bars\"></div><div class=\"small\">虚线 = 原图本身被认对的比例 " + pct(R.classifier.test_acc.value, 2) + "</div></div>";
    var b = el.querySelectorAll(".bars");
    bars(b[0], rows, "mse", mx, function (v) { return v.toFixed(4); });
    bars(b[1], rows, "recog_acc", 1, function (v) { return pct(v, 1); }, R.classifier.test_acc.value);
  }
  var panelsDone = { compress: false, invert: false };

  // 逐张对照 + 同样 10 个数
  var galSet = 0, galPred = {}, GALI = null, tenDigit = 7;
  var GAL_ROWS = [["orig", "原图", "", 784], ["inv_C1", "识别网络 · C1", "clf", 3136], ["inv_C2", "识别网络 · C2", "clf", 1568], ["inv_F1", "识别网络 · F1", "clf", 128], ["inv_logits", "识别网络 · 输出", "clf", 10]]
    .concat(DIMS.slice().reverse().map(function (d) { return ["ae_" + d, "自编码器 · " + d + " 维", "ae", d]; }));
  function galImgs(k) { if (k === "orig") return GAL.orig; GALI = GALI || {}; return GALI[k] || (GALI[k] = u8imgs(INV.gallery[k])); }
  function renderGallery() {
    var cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (c) { var n = 0; for (var i = 0; i < GAL.y.length; i++) if (GAL.y[i] === c && n++ === galSet) return i; return 0; });
    $("gal").innerHTML = GAL_ROWS.map(function (r) { return "<tr><th class=\"" + r[2] + "\">" + r[1] + "<small>" + r[3] + " 个数</small></th>" + cols.map(function () { return "<td><canvas class=\"px\"></canvas></td>"; }).join("") + "</tr>"; }).join("");
    Array.prototype.forEach.call($("gal").rows, function (tr, ri) {
      var key = GAL_ROWS[ri][0], imgs = galImgs(key);
      cols.forEach(function (i, ci) {
        var td = tr.cells[ci + 1], img = imgs[i], ck = key + ":" + i;
        ink(td.querySelector("canvas"), img);
        if (!(ck in galPred)) galPred[ck] = judge(img).pred;
        td.title = GAL_ROWS[ri][1] + "：真实数字 " + GAL.y[i] + "，识别网络认作 " + galPred[ck];
        if (galPred[ck] !== GAL.y[i]) { td.className = "miss"; td.insertAdjacentHTML("beforeend", "<span class=\"p\">" + galPred[ck] + "</span>"); }
      });
    });
  }
  function renderTen() {
    var idx = [];
    for (var i = 0; i < GAL.y.length; i++) if (GAL.y[i] === tenDigit) idx.push(i);
    var rows = [["orig", "原图", ""], ["inv_logits", "从识别网络输出（10 个数）还原", "clf"], ["ae_10", "自编码器 10 维还原", "ae"]];
    $("tenGrid").innerHTML = rows.map(function (r) { return "<div class=\"rl\" style=\"color:var(--" + (r[2] || "ink-2") + ")\">" + r[1] + "</div>" + idx.slice(0, 4).map(function () { return "<canvas class=\"px\"></canvas>"; }).join(""); }).join("");
    var cvs = $("tenGrid").querySelectorAll("canvas");
    rows.forEach(function (r, ri) { idx.slice(0, 4).forEach(function (i, ci) { ink(cvs[ri * 4 + ci], galImgs(r[0])[i]); }); });
    Array.prototype.forEach.call($("tenChips").children, function (b, c) { b.setAttribute("aria-pressed", c === tenDigit); });
  }
  function setupInvertStatic() {
    if (panelsDone.invert) return;
    panelsDone.invert = true;
    panels($("invPanels"), INV_ROWS);
    var iv = R.inversion, ae = R.autoencoder;
    $("invText").textContent = "从 C1 还原，误差只有 " + iv.C1.mse.value.toFixed(4) + "，几乎和原图一样；到了输出层，误差升到 " + iv.logits.mse.value.toFixed(4) +
      "，笔迹细节基本丢光，可是还原图仍有 " + pct(iv.logits.recog_acc.value, 1) + " 被认对。自编码器正相反：瓶颈越窄误差越大，2 维时被认对的只剩 " + pct(ae["2"].recog_acc.value, 1) + "。";
    for (var s = 0; s < 4; s++) {
      var b = document.createElement("button");
      b.className = "btn"; b.type = "button"; b.textContent = "第 " + (s + 1) + " 组"; b.dataset.s = s; b.setAttribute("aria-pressed", s === 0);
      b.onclick = function () { galSet = +this.dataset.s; $("galCtrl").querySelectorAll(".btn").forEach(function (x, j) { x.setAttribute("aria-pressed", j === galSet); }); renderGallery(); };
      $("galCtrl").appendChild(b);
    }
    $("tenChips").innerHTML = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (c) { return "<button class=\"chip\" type=\"button\">" + c + "</button>"; }).join("");
    Array.prototype.forEach.call($("tenChips").children, function (b, c) { b.onclick = function () { tenDigit = c; renderTen(); }; });
    var L = iv.logits, A = ae["10"];
    var row = function (name, a, b, fmt, lower) { var aw = lower ? a < b : a > b; return "<tr><td>" + name + "</td><td class=\"num" + (aw ? " win" : "") + "\">" + fmt(a) + "</td><td class=\"num" + (!aw ? " win" : "") + "\">" + fmt(b) + "</td></tr>"; };
    $("tenStats").innerHTML = "<thead><tr><th></th><th style=\"color:var(--clf)\">识别网络输出</th><th style=\"color:var(--ae)\">自编码器 10 维</th></tr></thead><tbody>" +
      row("像素误差 MSE（越小越像）", L.mse.value, A.mse.value, function (v) { return v.toFixed(4); }, true) +
      row("还原图被认对的比例", L.recog_acc.value, A.recog_acc.value, function (v) { return pct(v, 1); }, false) +
      row("只看编码认数字（5-NN）", L.code_knn5_acc.value, A.code_knn5_acc.value, function (v) { return pct(v, 1); }, false) + "</tbody>";
    $("tenText").textContent = "两边都是 10 个数。识别网络的 10 个数几乎只装着\"是几\"：从它还原出来的同一个数字，粗细、倾斜都被抹平成一个标准样子。10 维自编码器的 10 个数装着\"长什么样\"：每张图各有各的笔迹。\"只看编码认数字\"：把 10 个数当坐标，在训练集里找最近的 5 个邻居投票。";
    renderGallery();
    renderTen();
  }

  // ---------------- 大脑面板 ----------------
  var brainKey = null, brainSecs = [], K = 8;
  function brainSetup(key, secs) {
    if (key === brainKey) return;
    brainKey = key;
    brainSecs = secs;
    var body = $("brainBody");
    body.innerHTML = "";
    secs.forEach(function (s) {
      var el = document.createElement("div");
      el.className = "layer";
      el.innerHTML = "<div class=\"layer-title\"><b>" + s.title + "</b><span></span></div>";
      if (s.kind === "maps") el.insertAdjacentHTML("beforeend", "<div class=\"tiles\"></div><div class=\"strip\"></div>");
      if (s.kind === "dots") el.insertAdjacentHTML("beforeend", "<canvas class=\"dots\"></canvas>");
      if (s.kind === "code") el.insertAdjacentHTML("beforeend", "<div class=\"code-bars\"></div>");
      if (s.kind === "image") el.insertAdjacentHTML("beforeend", "<canvas class=\"px out-img\" width=\"28\" height=\"28\"></canvas>");
      if (s.kind === "probs") el.insertAdjacentHTML("beforeend", "<div class=\"probs\"></div>");
      body.appendChild(el);
      s.el = el;
      if (s.kind === "dots") {
        var cv = el.querySelector("canvas");
        cv.addEventListener("click", function (e) { var d = dotHit(s, e); if (d != null) toggleKo(s.model, s.tap, d); });
        cv.addEventListener("mousemove", function (e) {
          var d = dotHit(s, e), sp = el.querySelector(".layer-title span");
          sp.textContent = d == null ? s.note : "神经元 #" + d + "：激活 " + s.last.d[d].toFixed(2) + (isOff(s.model, s.tap, d) ? "（已敲掉）" : "") + "，点一下敲掉 / 恢复";
        });
      }
    });
  }
  function isOff(model, tap, i) { var m = M(model)[tap]; return !!(m && m.indexOf(i) >= 0); }
  function toggleKo(model, tap, i) {
    var m = M(model), a = m[tap] || (m[tap] = []), k = a.indexOf(i);
    if (k >= 0) a.splice(k, 1); else a.push(i);
    st.grid = null;
    refresh();
  }
  function refresh() {
    if (st.mode === "demo" || st.mode === "map") { if (st.x && st.zSource === "star") onInput(st.x, st.fromUser, false); else setZ(st.z, st.zSource === "demo" ? "demo" : "refresh"); }
    else if (st.mode === "compress") { var kn = st.knob; runCompress(); if (kn) { st.knob = kn; onKnob(); } }
    else if (st.mode === "invert") runInvert();
  }
  function chanMeans(t) {
    var n = t.s[0], per = t.d.length / n, m = new Float32Array(n);
    for (var c = 0; c < n; c++) { var s = 0; for (var i = 0; i < per; i++) s += t.d[c * per + i]; m[c] = s / per; }
    return m;
  }
  function drawMaps(s, t) {
    var n = t.s[0], H = t.s[1], per = H * t.s[2], means = chanMeans(t), k = Math.min(K, n), mx = 1e-6;
    var order = Array.from(means.keys()).sort(function (a, b) { return means[b] - means[a]; });
    for (var q = 0; q < t.d.length; q++) mx = Math.max(mx, t.d[q]);
    s.el.querySelector(".layer-title span").textContent = n + " 个通道，每个 " + H + "×" + t.s[2] + " · 展开最活跃的 " + k + " 个，下方小方块是全部 " + n + " 个（点击敲掉）";
    var tiles = s.el.querySelector(".tiles"), strip = s.el.querySelector(".strip");
    if (tiles.children.length !== k) {
      tiles.innerHTML = "";
      for (var i = 0; i < k; i++) {
        var b = document.createElement("button");
        b.type = "button"; b.className = "tile";
        b.innerHTML = "<div class=\"ko\"><canvas class=\"px\"></canvas></div><span></span>";
        b.onclick = function () { toggleKo(s.model, s.tap, +this.dataset.ch); };
        tiles.appendChild(b);
      }
    }
    Array.prototype.forEach.call(tiles.children, function (b, i) {
      var ch = order[i], img = new Float32Array(per);
      b.dataset.ch = ch;
      b.title = "通道 #" + ch + "：平均激活 " + means[ch].toFixed(2);
      b.querySelector(".ko").classList.toggle("off", isOff(s.model, s.tap, ch));
      b.querySelector("span").textContent = "#" + ch;
      for (var j = 0; j < per; j++) img[j] = t.d[ch * per + j] / mx;
      paintGray(b.querySelector("canvas"), img, t.s[2], H, C["--accent-deep"], C["--surface"]);
    });
    if (strip.children.length !== n) {
      strip.innerHTML = "";
      for (var c = 0; c < n; c++) {
        var sb = document.createElement("button");
        sb.type = "button"; sb.className = "ko"; sb.dataset.ch = c;
        sb.onclick = function () { toggleKo(s.model, s.tap, +this.dataset.ch); };
        strip.appendChild(sb);
      }
    }
    var mm = Math.max(1e-6, Math.max.apply(null, Array.from(means)));
    Array.prototype.forEach.call(strip.children, function (sb, c) {
      sb.style.background = rgb(mix(C["--surface"], C["--accent-deep"], means[c] / mm));
      sb.classList.toggle("off", isOff(s.model, s.tap, c));
      sb.title = "通道 #" + c + "：平均激活 " + means[c].toFixed(2);
      sb.setAttribute("aria-label", sb.title);
    });
  }
  function drawDots(s, t) {
    var cv = s.el.querySelector("canvas"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    var ctx = cv.getContext("2d"), n = t.d.length, cols = n >= 64 ? 16 : 8, rows = Math.ceil(n / cols), cw = w / cols, ch = h / rows, r = Math.max(2.5, Math.min(cw, ch) * 0.36);
    var mx = Math.max(1e-6, Math.max.apply(null, Array.from(t.d)));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    s.dots = [];
    for (var i = 0; i < n; i++) {
      var x = (i % cols + 0.5) * cw, y = (Math.floor(i / cols) + 0.5) * ch, off = isOff(s.model, s.tap, i);
      s.dots.push([x, y, r]);
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7);
      ctx.fillStyle = rgb(mix(C["--sunken"], C["--accent-deep"], off ? 0 : t.d[i] / mx)); ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = rgb(C["--line-strong"]); ctx.stroke();
      if (off) { ctx.strokeStyle = rgb(C["--bad"]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke(); }
    }
    s.note = n + " 个神经元（圆点越深越活跃），点一下敲掉";
    var sp = s.el.querySelector(".layer-title span");
    if (!sp.textContent || sp.textContent.indexOf("神经元 #") < 0) sp.textContent = s.note;
  }
  function dotHit(s, e) {
    if (!s.dots) return null;
    var r = s.el.querySelector("canvas").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = null, bd = 1e9;
    s.dots.forEach(function (d, i) { var dd = (d[0] - x) * (d[0] - x) + (d[1] - y) * (d[1] - y); if (dd <= (d[2] + 4) * (d[2] + 4) && dd < bd) { bd = dd; best = i; } });
    return best;
  }
  function drawCode(s, z, stats) {
    var box = s.el.querySelector(".code-bars"), n = z.length;
    s.el.querySelector(".layer-title span").textContent = n + " 个数" + (stats ? "（竖线 = 测试集平均值，条长 = 偏离多少个标准差）" : "");
    if (box.children.length !== n) box.innerHTML = Array.from({ length: n }, function (_, i) { return "<div class=\"code-bar\"><span>z" + (i + 1) + "</span><span class=\"track\"><i></i></span><b></b></div>"; }).join("");
    Array.prototype.forEach.call(box.children, function (row, i) {
      var v = z[i], m = stats ? stats.mean[i] : 0, sd = stats ? stats.std[i] || 1 : 1, u = Math.max(-3, Math.min(3, (v - m) / sd)) / 3 * 50;
      var bar = row.querySelector("i");
      bar.style.left = (u >= 0 ? 50 : 50 + u) + "%"; bar.style.width = Math.abs(u) + "%";
      row.querySelector("b").textContent = v.toFixed(2);
    });
  }
  function drawProbs(s, prob) {
    var box = s.el.querySelector(".probs"), p = NN.argmax(prob);
    s.el.querySelector(".layer-title span").textContent = "10 类的概率（最高：" + p + "，" + pct(prob[p]) + "）";
    box.innerHTML = prob.map(function (v, k) { return "<div class=\"" + (k === p ? "hi" : "") + "\"><i><b style=\"height:" + Math.max(2, v * 100) + "%\"></b></i>" + k + "</div>"; }).join("");
  }
  function maskCount(model) { var m = M(model), n = 0; Object.keys(m).forEach(function (k) { n += m[k].length; }); return n; }

  var brainModel = "ae_2";
  function renderBrain() {
    if (!st.ready && st.mode !== "demo") return;
    var mobileHidden = UI.isMobile() && document.body.classList.contains("op") && !document.body.classList.contains("brain-open");
    var title, formulas = [], params, secs, data = {};
    if ((st.mode === "compress" && !loaded.compress) || (st.mode === "invert" && !(loaded.compress && loaded.invert))) {
      brainKey = null;
      $("brainTitle").textContent = "网络内部";
      $("formula").textContent = ""; $("paramsLine").textContent = "";
      $("brainBody").innerHTML = "<p class=\"loading\">正在加载模型……</p>";
      return;
    }
    if (st.mode === "demo" || st.mode === "map") {
      var t = textAE(2);
      brainModel = "ae_2";
      title = "解码器怎样画出影子 · 2 维自编码器";
      formulas.push("解码器：" + t.dec);
      if (st.star && st.mode === "map") formulas.push("编码器（你写的数字 → ★）：" + t.enc);
      params = "参数 " + t.params.toLocaleString() + " 个（编码器 + 解码器）· 测试集平均像素误差 " + R.autoencoder["2"].mse.value.toFixed(4);
      brainSetup("dec2", [
        { kind: "code", title: "编码层（地图上的点）", model: "ae_2", tap: "z" },
        { kind: "dots", title: "全连接层", model: "ae_2", tap: "D1" },
        { kind: "maps", title: "变形成 32 张 7×7", model: "ae_2", tap: "D7" },
        { kind: "maps", title: "反卷积放大到 14×14", model: "ae_2", tap: "D14" },
        { kind: "image", title: "输出层：784 个数 = 影子", model: "ae_2", tap: "out" },
      ]);
      if (st.decRun) data = st.decRun.taps;
      data.stats = AE[2].stats;
    } else if (st.mode === "compress") {
      var d = st.selD, t2 = textAE(d);
      brainModel = "ae_" + d;
      title = d + " 维自编码器 · 压缩 → 还原";
      formulas.push("编码器：" + t2.enc, "解码器：" + t2.dec);
      params = "参数 " + t2.params.toLocaleString() + " 个 · 测试集平均像素误差 " + R.autoencoder[String(d)].mse.value.toFixed(4);
      brainSetup("ae" + d, [
        { kind: "maps", title: "编码器 · 卷积层 1", model: brainModel, tap: "C1", from: "enc" },
        { kind: "maps", title: "编码器 · 卷积层 2", model: brainModel, tap: "C2", from: "enc" },
        { kind: "code", title: "编码层（瓶颈）", model: brainModel, tap: "z", from: "dec" },
        { kind: "maps", title: "解码器 · 7×7", model: brainModel, tap: "D7", from: "dec" },
        { kind: "maps", title: "解码器 · 14×14", model: brainModel, tap: "D14", from: "dec" },
        { kind: "image", title: "输出层：还原图", model: brainModel, tap: "out", from: "dec" },
      ]);
      if (cmp && cmp[d]) { data = { enc: cmp[d].enc.taps, dec: cmp[d].dec.taps }; }
      data.stats = AE[d].stats;
    } else {
      title = "识别网络 · 训练好后冻结";
      brainModel = "classifier";
      formulas.push(ARCH.text(W, "classifier", CLF.layers, [1, 28, 28], "输入 784（28×28）", "输出 {n} 类"));
      params = "参数 " + NN.paramCount(W, "classifier", CLF.layers).toLocaleString() + " 个 · 测试集准确率 " + pct(R.classifier.test_acc.value, 2) + "。敲掉通道后，左边从这一层往后的还原图也会跟着变。";
      brainSetup("clf", [
        { kind: "maps", title: "卷积层 1（C1）", model: "classifier", tap: "C1" },
        { kind: "maps", title: "卷积层 2（C2）", model: "classifier", tap: "C2" },
        { kind: "dots", title: "全连接层（F1）", model: "classifier", tap: "F1" },
        { kind: "probs", title: "输出层", model: "classifier", tap: "out" },
      ]);
      if (inv) data = inv.clf.taps;
    }
    $("brainTitle").textContent = title;
    $("formula").innerHTML = formulas.map(function (f) { return "<div>" + f + "</div>"; }).join("");
    $("paramsLine").textContent = params;
    var n = maskCount(brainModel);
    $("koText").textContent = n ? "已敲掉 " + n + " 个通道 / 神经元" : "没有敲掉任何神经元。点特征图小方块或圆点可以敲掉它。";
    if (mobileHidden) return;
    renderArch();
    brainSecs.forEach(function (s) {
      var src = s.from ? data[s.from] : data, t = src && src[s.tap];
      if (!t) return;
      s.last = t;
      if (s.kind === "maps") drawMaps(s, t);
      else if (s.kind === "dots") drawDots(s, t);
      else if (s.kind === "code") drawCode(s, t.d, data.stats);
      else if (s.kind === "image") ink(s.el.querySelector("canvas"), t.d);
      else if (s.kind === "probs") drawProbs(s, NN.softmax(t.d));
    });
  }
  // ---------------- 结构图 ----------------
  var archView = new ARCHVIZ.View($("arch"), $("archInfo")), archNets = {};
  function aeNets(d) {
    return archNets["ae" + d] || (archNets["ae" + d] = {
      enc: ARCHVIZ.build(W, "ae_" + d, AE[d].encoder, [1, 28, 28], { codeLast: true }),
      dec: ARCHVIZ.build(W, "ae_" + d, AE[d].decoder, [d], {}),
    });
  }
  function renderArch() {
    var acts, note;
    if (st.mode === "demo" || st.mode === "map" || st.mode === "compress") {
      var d = st.mode === "compress" ? st.selD : 2, n = aeNets(d), m = M("ae_" + d), z, encActs;
      if (st.mode === "compress") { if (!cmp || !cmp[d]) return; z = cmp[d].zz; }
      else z = new Float32Array(st.z);
      if (st.x && (st.mode === "compress" || st.zSource === "star")) encActs = ARCHVIZ.run(n.enc, st.x, m).acts;
      else { encActs = n.enc.blocks.map(function () { return null; }); encActs[encActs.length - 1] = { d: Float32Array.from(z), s: [d] }; }
      acts = encActs.concat(ARCHVIZ.run(n.dec, z, m).acts.slice(1));
      archView.set("ae" + d, [{ net: n.enc, label: "编码器：784 个像素 → " + d + " 个数" }, { net: n.dec, from: 1, label: "解码器：" + d + " 个数 → 784 个像素（画出影子）" }]);
      note = d === 2 && st.mode !== "compress" ? "编码器把 784 个像素压成 2 个数，也就是地图上的一个点；解码器把这 2 个数画回 784 个像素。" + (st.zSource === "star" ? "" : "拖地图时没有用到编码器：点的位置直接就是编码，所以左半边是灰的。")
        : "沙漏形：编码器一路压缩到 " + d + " 个数，解码器再一路放大回 28×28。两半的卷积 / 反卷积大致对称。";
    } else {
      if (!inv) return;
      var net = archNets.clf || (archNets.clf = ARCHVIZ.build(W, "classifier", CLF.layers, [1, 28, 28], { outputLast: true }));
      acts = ARCHVIZ.run(net, st.x, M("classifier")).acts;
      archView.set("clf", [{ net: net, label: "第一课的识别网络（冻结）" }],
        [2, 4, 5, 6].map(function (at, i) { return { at: at, img: inv.items[i].out, label: "还原" }; }));
      note = "识别网络冻结不动；每条虚线是在那一层后面单独训练的反演解码器，下面是它从这一层还原出的图。越往后，还原得越模糊。";
    }
    $("archNote").textContent = note;
    archView.update(acts);
  }

  $("btnKoReset").onclick = function () { masks[brainModel] = {}; st.grid = null; refresh(); };
  $("btnKoRand").onclick = function () {
    var m = {};
    brainSecs.forEach(function (s) {
      if (s.kind !== "maps" && s.kind !== "dots") return;
      var t = s.last;
      if (!t) return;
      var n = t.s[0], idx = Array.from({ length: n }, function (_, i) { return i; });
      for (var i = n - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)), x = idx[i]; idx[i] = idx[j]; idx[j] = x; }
      m[s.tap] = idx.slice(0, Math.round(n * 0.25));
    });
    masks[brainModel] = m;
    refresh();
  };

  // ---------------- 练习任务（数字地图） ----------------
  var TASKS = [
    { id: "digits", short: "0–9 的影子", long: "在地图上找到 0–9 每个数字的影子（识别网络把握 ≥ 80%）" },
    { id: "odd", short: "谁都不像", long: "找一个\"谁都不像\"的影子：识别网络最高把握也不到 50%" },
    { id: "star", short: "写一个看落点", long: "在小画板上写一个数字，看它落在地图上哪里（★）" },
    { id: "interp", short: "播放渐变", long: "播放一次\"从一个数字走到另一个数字\"" },
  ];
  function done(id) { return id === "digits" ? Object.keys(st.tasks.digits).length === 10 : !!st.tasks[id]; }
  function renderTasks() {
    var n = 0;
    $("tasks").innerHTML = TASKS.map(function (t) {
      var ok = done(t.id);
      if (ok) n++;
      var extra = t.id === "digits" ? "<div class=\"digit-chips\">" + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (d) { return "<span class=\"" + (st.tasks.digits[d] ? "on" : "") + "\">" + d + "</span>"; }).join("") + "</div>" : "";
      var cnt = t.id === "digits" ? "（" + Object.keys(st.tasks.digits).length + "/10）" : "";
      return "<li class=\"" + (ok ? "done" : "") + "\"><span class=\"box\">" + (ok ? "✓" : "") + "</span><span class=\"txt\"><span class=\"long\">" + t.long + cnt + "</span><span class=\"short\">" + t.short + cnt + "</span>" + extra + "</span></li>";
    }).join("");
    $("taskCount").textContent = n === TASKS.length ? "全部完成！" : "已完成 " + n + " / " + TASKS.length;
  }

  // ---------------- 模式 ----------------
  var INTRO = {
    demo: ["这一页在演示什么", "左边的地图上，每个小点是一张测试集图片被压缩后的位置（只剩 2 个数），颜色是它的真实数字。橙色的点在地图上自动游走，每到一处，解码器就把这 2 个数画成一张图：这就是这个点的\"影子\"。地图上没有图片的空白处，也能画出影子。"],
    map: ["怎么玩", "拖动地图上的点看影子怎么变；在小画板上写一个数字，它会被压成一个 ★ 落到地图上。0、1、6 这样的数字各自成团；4 与 9、3/5/8 挤在一起，影子也最容易画混。"],
    compress: ["压缩得越狠，丢得越多", "同一套卷积结构，瓶颈分别是 2、4、10、16、32 个数。瓶颈越窄，还原图越模糊，最先丢掉的是笔迹的粗细、倾斜等细节，最后连\"是几\"也保不住。拧旋钮可以直接改编码里的数，看每个数大致管着什么。"],
    invert: ["识别网络只记得\"是几\"", "识别网络训练时只因\"认错\"受罚。越往深层，粗细、倾斜这些和\"是几\"无关的信息被丢得越多；到了输出层，只剩 10 个数，从它还原出来的同一个数字几乎长一个样。同样 10 个数，自编码器记住的是\"长什么样\"。"],
  };
  function setMode(id) {
    st.mode = id;
    cancelAnimationFrame(st.demoRaf);
    stopInterp();
    placePad();
    $("demoBadge").hidden = id !== "demo";
    $("cardIntro").innerHTML = "<h2>" + INTRO[id][0] + "</h2><p>" + INTRO[id][1] + "</p>";
    if (id === "demo" || id === "map") {
      brainKey = null;
      requestAnimationFrame(function () {
        drawMapBase();
        if (id === "demo") { st.lastFrame = 0; st.demoRaf = requestAnimationFrame(demoLoop); }
        else setZ(st.z, "refresh");
      });
    } else if (id === "compress") {
      brainKey = null;
      $("reconGrid").innerHTML = "<p class=\"loading\">正在加载 4 / 10 / 16 / 32 维自编码器……</p>";
      need(["compress"], function () {
        if (!panelsDone.compress) { panelsDone.compress = true; panels($("aePanels"), AE_ROWS); }
        ensureInput(); runCompress();
      });
    } else if (id === "invert") {
      brainKey = null;
      $("invGrid").innerHTML = "<p class=\"loading\">正在加载反演解码器……</p>";
      need(["compress", "invert"], function () { setupInvertStatic(); ensureInput(); runInvert(); });
    }
    renderBrain();
  }

  // ---------------- 控件 ----------------
  $("digitChips").innerHTML = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (c) { return "<button class=\"chip\" type=\"button\" aria-pressed=\"false\" title=\"只看数字 " + c + "\"><i style=\"background:var(--c" + c + ")\"></i>" + c + "</button>"; }).join("");
  Array.prototype.forEach.call($("digitChips").children, function (b, c) {
    b.onclick = function () { st.focus = st.focus === c ? null : c; Array.prototype.forEach.call($("digitChips").children, function (x, j) { x.setAttribute("aria-pressed", st.focus === j); }); drawMapBase(); };
  });
  var setMapMode = function (m) { st.mapMode = m; $("btnPts").setAttribute("aria-pressed", m === "pts"); $("btnGrid").setAttribute("aria-pressed", m === "grid"); drawMapBase(); };
  var setView = function (v) { st.view = v; st.grid = null; $("btnZoom").setAttribute("aria-pressed", v === "zoom"); $("btnFull").setAttribute("aria-pressed", v === "full"); drawMapBase(); };
  $("btnPts").onclick = function () { setMapMode("pts"); };
  $("btnGrid").onclick = function () { setMapMode("grid"); };
  $("btnZoom").onclick = function () { setView("zoom"); };
  $("btnFull").onclick = function () { setView("full"); };
  ["selA", "selB"].forEach(function (id, k) { $(id).innerHTML = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (c) { return "<option value=\"" + c + "\">" + c + "</option>"; }).join(""); $(id).value = k ? 7 : 0; });
  $("btnPlay").onclick = playInterp;

  readColors();
  UI.addStopper(function () { cancelAnimationFrame(st.demoRaf); stopInterp(); W = null; });
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { readColors(); st.grid = null; pad.redraw(); if (UI.current().lesson === "shadow") setMode(st.mode); }); } catch (e) { /* 旧浏览器 */ }
  st.ready = true;
  setZ(routeAt(0), "demo");
  renderTasks();

  // 供自动化验收用（tools/browser_check）
  window.MAP_TEST = {
    state: function () {
      var dd = st.mode === "compress" ? st.selD : 2, ready = st.mode === "compress" ? loaded.compress : st.mode === "invert" ? loaded.invert : true;
      var t = st.mode === "invert" || !ready ? null : textAE(dd), py = AE[dd];
      return {
        ready: st.ready, mode: st.mode, help: UI.helpOpen(), mobile: UI.isMobile(), brainOpen: document.body.classList.contains("brain-open"),
        z: st.z, zSource: st.zSource, shadowPred: st.judge && st.judge.pred, shadowConf: st.judge && st.judge.conf, star: st.star, hasInput: !!st.x,
        tasks: TASKS.map(function (x) { return [x.id, done(x.id)]; }), digitsFound: Object.keys(st.tasks.digits).length, loaded: loaded,
        brainKey: brainKey, brainModel: brainModel, knocked: maskCount(brainModel), selD: st.selD, knob: !!st.knob,
        reconCards: $("reconGrid").children.length, invCards: $("invGrid").children.length, galCells: $("gal").querySelectorAll("canvas").length,
        formulaOk: !ready ? null : t ? t.enc === py.encoder_text && t.dec === py.decoder_text && t.params === py.params
          : ARCH.text(W, "classifier", CLF.layers, [1, 28, 28], "输入 784（28×28）", "输出 {n} 类") === CLF.text && NN.paramCount(W, "classifier", CLF.layers) === CLF.params,
        view: st.view, mapMode: st.mapMode,
      };
    },
    zToClient: function (zx, zy) { var S = mapSize(), p = toPx(zx, zy, S); return [S.left + p[0], S.top + p[1]]; },
    centers: CENTERS,
  };

  return {
    setMode: setMode,
    leave: function () { cancelAnimationFrame(st.demoRaf); stopInterp(); },
    onLayout: function () { if (st.mode === "demo" || st.mode === "map") drawMapBase(); brainKey = null; renderBrain(); },
    onBrain: function (open) { if (open) requestAnimationFrame(function () { brainKey = null; renderBrain(); }); },
    toggleGrid: function () { setMapMode(st.mapMode === "grid" ? "pts" : "grid"); },
    toggleView: function () { setView(st.view === "zoom" ? "full" : "zoom"); },
    onKey: function (e) {
      var k = e.key.toLowerCase();
      if (k === "g") setMapMode(st.mapMode === "grid" ? "pts" : "grid");
      else if (k === "f") setView(st.view === "zoom" ? "full" : "zoom");
      else if (k === "p" && st.mode === "map") playInterp();
      else if (k === "c") $("btnClear").click();
      else if (k === "z") $("btnUndo").click();
      else if (k === "e") $("btnSample").click();
      else if (st.mode === "compress" && k >= "1" && k <= String(DIMS.length)) { st.selD = DIMS[+k - 1]; st.knob = null; runCompress(); }
    },
  };
  }
})();
