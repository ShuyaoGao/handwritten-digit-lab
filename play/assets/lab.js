// 第一课 · 识别：画板 → 预处理 → 三档网络（纯 JS 推理）→ 大脑面板（结构图 + 每层激活）
(function () {
  "use strict";
  var D = window.DIGIT_LAB_DATA;
  var W = NN.loadWeights(D.weights.manifest, D.weights.b64);
  var $ = function (id) { return document.getElementById(id); };
  var PRESETS = D.presets;
  var byId = {};
  PRESETS.forEach(function (p) { byId[p.id] = p; p.formula = ARCH.classifierText(W, p.id, p.layers); p.nParams = NN.paramCount(W, p.id, p.layers); p.taps = ARCH.tapShapes(W, p.id, p.layers); });
  var SHORT = { mlp: "全连接", cnn_s: "小卷积", cnn_m: "标准卷积" };

  // 测试集示例图
  var SAMPLES = (function () {
    var s = atob(D.samples.b64), out = [];
    for (var i = 0; i < D.samples.n; i++) {
      var a = new Float32Array(784);
      for (var j = 0; j < 784; j++) a[j] = s.charCodeAt(i * 784 + j) / 255;
      out.push(a);
    }
    return out;
  })();

  var st = {
    ready: false, mode: "demo", preset: "cnn_m", masks: { mlp: {}, cnn_s: {}, cnn_m: {} },
    x: null, fromUser: false, sample: -1, res: null, base: null, all: {}, heat: null, heatFor: null, heatToken: 0,
    K: 8, showHeat: true, demoPaused: false, demoTimer: null, demoI: 0,
    tasks: { digits: {}, hesitate: false, knock: false, disagree: false },
  };
  var C = {};
  function readColors() {
    ["--surface", "--sunken", "--ink", "--ink-2", "--muted", "--line", "--line-strong", "--accent", "--accent-deep", "--pos", "--neg", "--bad"].forEach(function (k) { C[k] = cssColor(k); });
  }
  var rgb = function (c, a) { return "rgba(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + "," + (a == null ? 1 : a) + ")"; };
  var mix = function (a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; };
  var pct = function (v) { return Math.round(v * 100) + "%"; };

  // ---------------- 画板 ----------------
  var pad = new Pad($("pad"), {
    lineWidth: 22,
    onDraw: function () { useInput(pad.toInput(), true, false); },
    onEnd: function () { useInput(pad.toInput(), true, true); },
  });
  // 演示模式下一落笔就切到「我来写」
  $("padBox").addEventListener("pointerdown", function () { if (st.mode === "demo") UI.setTab("draw"); $("padHint").hidden = true; }, true);

  function loadSample(i, demo) {
    st.sample = (i + SAMPLES.length) % SAMPLES.length;
    pad.setSample(SAMPLES[st.sample]);
    $("padHint").hidden = true;
    var b = $("demoBadge");
    b.hidden = !demo;
    if (demo) b.textContent = "演示 · 测试集样本 · 真实答案 " + D.samples.y[st.sample];
    useInput(SAMPLES[st.sample], false, true);
  }

  // ---------------- 推理 ----------------
  function maskCount(id) {
    var m = st.masks[id], n = 0;
    Object.keys(m).forEach(function (k) { n += m[k].length; });
    return n;
  }
  function compute() {
    var x = st.x || new Float32Array(784), p = byId[st.preset];
    st.res = NN.run(W, p.id, p.layers, x, { masks: st.masks[p.id] });
    st.res.prob = NN.softmax(st.res.out);
    st.res.pred = NN.argmax(st.res.prob);
    st.base = maskCount(p.id) ? NN.argmax(NN.run(W, p.id, p.layers, x).out) : st.res.pred;
    PRESETS.forEach(function (q) { st.all[q.id] = q.id === p.id && !maskCount(p.id) ? st.res.pred : NN.argmax(NN.run(W, q.id, q.layers, x).out); });
  }

  function useInput(x, fromUser, final) {
    st.x = x;
    st.fromUser = fromUser;
    if (fromUser) $("demoBadge").hidden = true;
    compute();
    if (!final) { st.heat = null; render(); return; }
    checkTasks();
    render();
    startHeat();
  }

  // 遮挡法热力图：依次遮住 5×5 的小块，看对预测类别的把握掉多少（只遮有笔迹的地方）
  function startHeat() {
    var token = ++st.heatToken, x = st.x, p = byId[st.preset];
    st.heat = null;
    if (!x || !st.showHeat) { renderHeat(); return; }
    var masks = st.masks[p.id], cls = st.res.pred, p0 = st.res.prob[cls];
    var acc = new Float32Array(784), cnt = new Float32Array(784), jobs = [];
    for (var r = 0; r <= 22; r += 2) for (var c = 0; c <= 22; c += 2) jobs.push([r, c]);
    var k = 0;
    (function step() {
      if (token !== st.heatToken || UI.stopped()) return;
      var t0 = performance.now();
      while (k < jobs.length && performance.now() - t0 < 12) {
        var r0 = jobs[k][0], c0 = jobs[k][1], ink = 0, xo = Float32Array.from(x), i, j;
        for (i = r0; i < r0 + 5; i++) for (j = c0; j < c0 + 5; j++) { ink += x[i * 28 + j]; xo[i * 28 + j] = 0; }
        var drop = 0;
        // 用对数概率的下降量：网络很有把握时，概率本身几乎不动，对数概率仍然灵敏
        if (ink > 0.05) drop = Math.max(0, Math.log(p0) - Math.log(Math.max(1e-12, NN.softmax(NN.run(W, p.id, p.layers, xo, { masks: masks }).out)[cls])));
        for (i = r0; i < r0 + 5; i++) for (j = c0; j < c0 + 5; j++) { acc[i * 28 + j] += drop; cnt[i * 28 + j] += 1; }
        k++;
      }
      if (k < jobs.length) { setTimeout(step, 0); return; }
      var mx = 1e-6;
      for (var q = 0; q < 784; q++) { acc[q] = cnt[q] ? acc[q] / cnt[q] : 0; mx = Math.max(mx, acc[q]); }
      for (var q2 = 0; q2 < 784; q2++) acc[q2] /= mx;
      st.heat = mx > 1e-4 ? acc : new Float32Array(784);
      st.heatFor = cls;
      renderHeat();
    })();
  }

  // ---------------- 练习任务 ----------------
  var TASKS = [
    { id: "digits", short: "0–9 都认对", long: "让网络把你亲手写的 0–9 每个数字都认出来（把握 ≥ 80%）" },
    { id: "hesitate", short: "让它犹豫", long: "写一个让网络犹豫的数字：最高的把握也不到 60%" },
    { id: "knock", short: "敲掉改判", long: "在右边敲掉通道或神经元，让网络改变判断" },
    { id: "disagree", short: "三档不一致", long: "写一个三档网络判断不一样的数字（看\"网络的判断\"下面那一行）" },
  ];
  function checkTasks() {
    if (!st.x || !st.res) return;
    var conf = st.res.prob[st.res.pred], t = st.tasks;
    if (st.fromUser && !maskCount(st.preset)) {
      if (conf >= 0.8) t.digits[st.res.pred] = true;
      if (conf < 0.6) t.hesitate = true;
      var ps = PRESETS.map(function (q) { return st.all[q.id]; });
      if (ps.some(function (v) { return v !== ps[0]; })) t.disagree = true;
    }
    if (maskCount(st.preset) && st.res.pred !== st.base) t.knock = true;
  }
  function taskDone(id) { return id === "digits" ? Object.keys(st.tasks.digits).length === 10 : st.tasks[id]; }
  function renderTasks() {
    var n = 0;
    $("tasks").innerHTML = TASKS.map(function (t) {
      var done = taskDone(t.id);
      if (done) n++;
      var extra = t.id === "digits" ? "<div class=\"digit-chips\">" + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (d) { return "<span class=\"" + (st.tasks.digits[d] ? "on" : "") + "\">" + d + "</span>"; }).join("") + "</div>" : "";
      return "<li class=\"" + (done ? "done" : "") + "\" data-task=\"" + t.id + "\"><span class=\"box\">" + (done ? "✓" : "") + "</span><span class=\"txt\"><span class=\"long\">" + t.long + "</span><span class=\"short\">" + t.short + "</span>" + extra + "</span></li>";
    }).join("");
    $("taskCount").textContent = n === TASKS.length ? "全部完成！" : "已完成 " + n + " / " + TASKS.length;
  }

  // ---------------- 绘制 ----------------
  function chanMeans(t) {
    var Cn = t.s[0], per = t.d.length / Cn, m = new Float32Array(Cn);
    for (var c = 0; c < Cn; c++) { var s = 0; for (var i = 0; i < per; i++) s += t.d[c * per + i]; m[c] = s / per; }
    return m;
  }
  function isOff(layer, i) { var m = st.masks[st.preset][layer]; return !!(m && m.indexOf(i) >= 0); }
  function toggleKo(layer, i) {
    var m = st.masks[st.preset], a = m[layer] || (m[layer] = []), k = a.indexOf(i);
    if (k >= 0) a.splice(k, 1); else a.push(i);
    useInput(st.x, st.fromUser, true);
  }

  function renderConv(layer, title, tilesEl, stripEl, label) {
    var t = st.res.taps[layer], Cn = t.s[0], H = t.s[1], per = H * H, means = chanMeans(t);
    var order = Array.from(means.keys()).sort(function (a, b) { return means[b] - means[a]; });
    var K = Math.min(st.K, Cn), mx = 1e-6;
    for (var q = 0; q < t.d.length; q++) mx = Math.max(mx, t.d[q]);
    title.textContent = Cn + " 个通道，每个 " + H + "×" + H + "（" + label + "）· 展开最活跃的 " + K + " 个，下方小方块是全部 " + Cn + " 个";
    if (tilesEl.children.length !== K) {
      tilesEl.innerHTML = "";
      for (var i = 0; i < K; i++) {
        var b = document.createElement("button");
        b.type = "button"; b.className = "tile";
        b.innerHTML = "<div class=\"ko\"><canvas class=\"px\"></canvas></div><span></span>";
        b.onclick = function () { toggleKo(layer, +this.dataset.ch); };
        tilesEl.appendChild(b);
      }
    }
    Array.prototype.forEach.call(tilesEl.children, function (b, i) {
      var ch = order[i];
      b.dataset.ch = ch;
      b.title = "通道 #" + ch + "：平均激活 " + means[ch].toFixed(2) + "（点一下敲掉 / 恢复）";
      b.querySelector(".ko").classList.toggle("off", isOff(layer, ch));
      b.querySelector("span").textContent = "#" + ch;
      var img = new Float32Array(per);
      for (var j = 0; j < per; j++) img[j] = t.d[ch * per + j] / mx;
      paintGray(b.querySelector("canvas"), img, H, H, C["--accent-deep"], C["--surface"]);
    });
    if (stripEl.children.length !== Cn) {
      stripEl.innerHTML = "";
      for (var c = 0; c < Cn; c++) {
        var s = document.createElement("button");
        s.type = "button"; s.className = "ko"; s.dataset.ch = c;
        s.onclick = function () { toggleKo(layer, +this.dataset.ch); };
        stripEl.appendChild(s);
      }
    }
    var mm = Math.max(1e-6, Math.max.apply(null, Array.from(means)));
    Array.prototype.forEach.call(stripEl.children, function (s, c) {
      s.style.background = rgb(mix(C["--surface"], C["--accent-deep"], means[c] / mm));
      s.classList.toggle("off", isOff(layer, c));
      s.title = "通道 #" + c + "：平均激活 " + means[c].toFixed(2);
      s.setAttribute("aria-label", s.title);
    });
  }

  var fcDots = [];
  function renderFC() {
    var cv = $("fcCanvas"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    var ctx = cv.getContext("2d"), p = byId[st.preset];
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var a = st.res.taps.F1.d, n = a.length, cols = n >= 64 ? 16 : 8, rows = Math.ceil(n / cols);
    var leftW = w * 0.56, cw = leftW / cols, ch = (h - 10) / rows, r = Math.max(2.5, Math.min(cw, ch) * 0.36);
    var mx = Math.max(1e-6, Math.max.apply(null, Array.from(a)));
    fcDots = [];
    for (var i = 0; i < n; i++) fcDots.push({ i: i, x: (i % cols + 0.5) * cw, y: 5 + (Math.floor(i / cols) + 0.5) * ch, r: r });
    // 输出层
    var ox = leftW + 34, ow = w - ox - 44, rh = (h - 10) / 10, pred = st.res.pred, prob = st.res.prob;
    // 连线：对预测类别贡献最大的 8 个神经元（w × 激活）
    var w2 = W[p.id + ".fc2.weight"].d, contrib = [];
    for (var j = 0; j < n; j++) contrib.push([j, w2[pred * n + j] * a[j]]);
    contrib.sort(function (u, v) { return Math.abs(v[1]) - Math.abs(u[1]); });
    var top = contrib.slice(0, 8).filter(function (c) { return Math.abs(c[1]) > 1e-6; }), cm = top.length ? Math.abs(top[0][1]) : 1;
    var ty = 5 + (pred + 0.5) * rh;
    top.forEach(function (c) {
      var d = fcDots[c[0]];
      ctx.strokeStyle = rgb(c[1] > 0 ? C["--pos"] : C["--neg"], 0.75);
      ctx.lineWidth = 1 + 3 * Math.abs(c[1]) / cm;
      ctx.beginPath(); ctx.moveTo(d.x, d.y);
      ctx.bezierCurveTo(d.x + 60, d.y, ox - 70, ty, ox - 22, ty); ctx.stroke();
    });
    fcDots.forEach(function (d) {
      var off = isOff("F1", d.i);
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7);
      ctx.fillStyle = rgb(mix(C["--sunken"], C["--accent-deep"], off ? 0 : a[d.i] / mx)); ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = rgb(C["--line-strong"]); ctx.stroke();
      if (off) {
        ctx.strokeStyle = rgb(C["--bad"]); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(d.x - d.r, d.y - d.r); ctx.lineTo(d.x + d.r, d.y + d.r); ctx.moveTo(d.x + d.r, d.y - d.r); ctx.lineTo(d.x - d.r, d.y + d.r); ctx.stroke();
      }
    });
    ctx.font = "600 13px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
    ctx.textBaseline = "middle";
    for (var k = 0; k < 10; k++) {
      var y = 5 + (k + 0.5) * rh, bh = Math.min(16, rh * 0.62);
      ctx.fillStyle = rgb(k === pred ? C["--ink"] : C["--muted"]); ctx.textAlign = "right"; ctx.fillText(k, ox - 8, y);
      ctx.fillStyle = rgb(C["--sunken"]); ctx.fillRect(ox, y - bh / 2, ow, bh);
      ctx.fillStyle = rgb(k === pred ? C["--accent"] : C["--line-strong"]); ctx.fillRect(ox, y - bh / 2, Math.max(1, ow * prob[k]), bh);
      ctx.fillStyle = rgb(k === pred ? C["--ink"] : C["--muted"]); ctx.textAlign = "left"; ctx.font = "12px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
      ctx.fillText(pct(prob[k]), ox + ow + 6, y);
      ctx.font = "600 13px " + getComputedStyle(document.body).getPropertyValue("--f-mono");
    }
    $("fcTitle").textContent = n + " 个神经元（圆点越深越活跃）→ 10 类的概率；连线是对\"" + pred + "\"贡献最大的 " + top.length + " 个（橙 = 推高，蓝 = 压低）";
  }
  function fcHit(e) {
    var r = $("fcCanvas").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = null;
    fcDots.forEach(function (d) { var dd = (d.x - x) * (d.x - x) + (d.y - y) * (d.y - y); if (dd <= (d.r + 4) * (d.r + 4) && (!best || dd < best.dd)) best = { d: d, dd: dd }; });
    return best && best.d;
  }
  $("fcCanvas").addEventListener("mousemove", function (e) {
    var d = fcHit(e), p = byId[st.preset];
    if (!d || !st.res) { $("fcHover").textContent = "鼠标移到圆点上看单个神经元；点一下把它\"敲掉\"。"; return; }
    var a = st.res.taps.F1.d[d.i], n = st.res.taps.F1.d.length, c = W[p.id + ".fc2.weight"].d[st.res.pred * n + d.i] * a;
    $("fcHover").textContent = "神经元 #" + d.i + "：激活 " + a.toFixed(2) + " · 对\"" + st.res.pred + "\"的贡献 " + (c >= 0 ? "+" : "") + c.toFixed(2) + (isOff("F1", d.i) ? "（已敲掉）" : "");
  });
  $("fcCanvas").addEventListener("click", function (e) { var d = fcHit(e); if (d) toggleKo("F1", d.i); });

  function renderHeat() {
    var cv = $("heatCanvas"), ctx = cv.getContext("2d"), im = ctx.createImageData(28, 28), h = st.heat;
    if (h && st.showHeat) {
      for (var i = 0; i < 784; i++) { im.data[4 * i] = C["--pos"][0]; im.data[4 * i + 1] = C["--pos"][1]; im.data[4 * i + 2] = C["--pos"][2]; im.data[4 * i + 3] = Math.round(h[i] * 200); }
    }
    ctx.putImageData(im, 0, 0);
    $("heatNote").textContent = !st.showHeat ? "热力图已关闭（左侧\"细调\"里可以打开）。"
      : !st.x ? "写一个数字后，这里会叠加\"网络在看哪里\"。"
      : !h ? "正在计算\"网络在看哪里\"……"
      : "橙色越深 = 遮住这一小块后，网络对\"" + st.heatFor + "\"的把握掉得越多。这是相关性解释，不代表因果。";
  }

  function render() {
    if (!st.res) return;
    var p = byId[st.preset], x = st.x || new Float32Array(784), has = !!st.x;
    paintGray($("seen"), x, 28, 28, C["--ink"], C["--surface"]);
    paintGray($("inCanvas"), x, 28, 28, C["--ink"], C["--surface"]);
    var pred = st.res.pred, prob = st.res.prob;
    $("predDigit").textContent = has ? pred : "–";
    $("predConf").textContent = has ? "把握 " + pct(prob[pred]) : "先写一个数字";
    $("miniProbs").innerHTML = prob.map(function (v, k) { return "<div class=\"" + (has && k === pred ? "hi" : "") + "\"><i><b style=\"height:" + (has ? Math.max(2, v * 100) : 0) + "%\"></b></i>" + k + "</div>"; }).join("");
    $("predAll").textContent = has ? "三档网络：" + PRESETS.map(function (q) { return SHORT[q.id] + " " + st.all[q.id]; }).join(" · ") + (maskCount(p.id) ? "（" + SHORT[p.id] + "已敲掉神经元）" : "") : "";
    $("presetName").textContent = p.name;
    $("formula").textContent = p.formula;
    $("paramsLine").textContent = "参数 " + p.nParams.toLocaleString() + " 个 · 测试集准确率 " + (p.test_acc * 100).toFixed(2) + "%（MNIST 测试集 1 万张）";
    var conv = !!p.taps.C1;
    $("layerC1").hidden = $("layerC2").hidden = !conv;
    $("noConv").hidden = conv;
    if (conv) {
      renderConv("C1", $("c1Title"), $("c1Tiles"), $("c1Strip"), "卷积 + 池化后");
      renderConv("C2", $("c2Title"), $("c2Tiles"), $("c2Strip"), "卷积 + 池化后");
    }
    renderFC();
    renderHeat();
    renderArch();
    var n = maskCount(p.id);
    $("koText").textContent = n ? "已敲掉 " + n + " 个（" + Object.keys(st.masks[p.id]).filter(function (k) { return st.masks[p.id][k].length; }).map(function (k) { return { C1: "卷积层 1", C2: "卷积层 2", F1: "全连接" }[k] + " " + st.masks[p.id][k].length; }).join("，") + "）" + (has && pred !== st.base ? "：判断从 " + st.base + " 变成了 " + pred : "") : "没有敲掉任何神经元。点特征图小方块或圆点可以敲掉它。";
    document.querySelectorAll(".preset").forEach(function (b) { b.setAttribute("aria-checked", b.dataset.id === p.id); });
    renderTasks();
  }

  // ---------------- 结构图 ----------------
  var arch = new ARCHVIZ.View($("archRec"), $("archRecInfo")), archNets = {};
  function renderArch() {
    if (UI.isMobile() && document.body.classList.contains("op") && !document.body.classList.contains("brain-open")) return;
    var p = byId[st.preset];
    var net = archNets[p.id] || (archNets[p.id] = ARCHVIZ.build(W, p.id, p.layers, [1, 28, 28], { outputLast: true }));
    arch.set(p.id, [{ net: net, label: "识别网络 · " + p.name }]);
    arch.update(ARCHVIZ.run(net, st.x || new Float32Array(784), st.masks[p.id]).acts);
  }

  // ---------------- 控件 ----------------
  $("presets").innerHTML = PRESETS.map(function (p, i) {
    return "<button type=\"button\" class=\"preset\" role=\"radio\" data-id=\"" + p.id + "\" title=\"" + p.formula + "（快捷键 " + (i + 1) + "）\"><b>" + p.name + "</b><span>参数 " + p.nParams.toLocaleString() + "<br>准确率 " + (p.test_acc * 100).toFixed(2) + "%</span></button>";
  }).join("");
  document.querySelectorAll(".preset").forEach(function (b) { b.onclick = function () { setPreset(b.dataset.id); }; });
  function setPreset(id) { st.preset = id; useInput(st.x, st.fromUser, true); }

  $("btnClear").onclick = function () { pad.clear(); $("padHint").hidden = false; $("demoBadge").hidden = true; if (st.mode === "demo") UI.setTab("draw"); useInput(null, true, true); };
  $("btnUndo").onclick = function () { pad.undo(); useInput(pad.toInput(), true, true); };
  $("btnSample").onclick = function () { loadSample(Math.floor(Math.random() * SAMPLES.length), false); };
  $("btnPause").onclick = function () { st.demoPaused = !st.demoPaused; $("btnPause").setAttribute("aria-pressed", st.demoPaused); $("btnPause").textContent = st.demoPaused ? "继续" : "暂停"; };
  $("btnNext").onclick = function () { st.demoI++; loadSample(st.demoI, true); };
  $("rngWidth").oninput = function () { pad.setLineWidth(+this.value); $("valWidth").textContent = this.value; if (!pad.isEmpty() && !pad.sample) useInput(pad.toInput(), true, true); };
  $("rngK").oninput = function () { st.K = +this.value; $("valK").textContent = this.value; render(); };
  $("chkHeat").onchange = function () { st.showHeat = this.checked; startHeat(); };
  $("btnKoReset").onclick = function () { st.masks[st.preset] = {}; useInput(st.x, st.fromUser, true); };
  $("btnKoRand").onclick = function () {
    var p = byId[st.preset], m = {};
    Object.keys(p.taps).filter(function (k) { return k !== "out"; }).forEach(function (k) {
      var n = p.taps[k][0], idx = Array.from({ length: n }, function (_, i) { return i; });
      for (var i = n - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
      m[k] = idx.slice(0, Math.round(n * 0.25));
    });
    st.masks[p.id] = m;
    useInput(st.x, st.fromUser, true);
  };

  function demoTick() { if (st.mode === "demo" && !st.demoPaused && !UI.helpOpen()) { st.demoI++; loadSample(st.demoI, true); } }
  function setMode(id) {
    st.mode = id;
    clearInterval(st.demoTimer);
    st.demoTimer = null;
    if (id === "demo") { $("padHint").hidden = true; loadSample(st.demoI, true); st.demoTimer = setInterval(demoTick, 2800); }
    else { $("demoBadge").hidden = true; if (pad.isEmpty()) $("padHint").hidden = false; }
    render();
    requestAnimationFrame(function () { renderFC(); });
  }

  readColors();
  UI.lesson({
    id: "rec",
    name: "第一课 · 识别",
    short: "识别",
    tabs: [{ id: "demo", name: "演示", title: "自动轮播测试集里的手写数字" }, { id: "draw", name: "我来写", short: "写", op: true }, { id: "practice", name: "练习", op: true }],
    onTab: setMode,
    onLeave: function () { clearInterval(st.demoTimer); st.demoTimer = null; st.heatToken++; },
    onLayout: function () { if (st.res) { renderFC(); renderArch(); } },
    onBrain: function (open) { if (open) requestAnimationFrame(function () { render(); }); },
    menuItems: PRESETS.map(function (p) { return ["网络：" + p.name, function () { setPreset(p.id); }]; }).concat([["热力图：开 / 关", function () { $("chkHeat").checked = !$("chkHeat").checked; $("chkHeat").onchange.call($("chkHeat")); }]]),
    onKey: function (e) {
      var k = e.key.toLowerCase();
      if (k === "c") $("btnClear").click();
      else if (k === "z") $("btnUndo").click();
      else if (k === "e") $("btnSample").click();
      else if (k === "n" && st.mode === "demo") $("btnNext").click();
      else if (k === " " && st.mode === "demo") { $("btnPause").click(); e.preventDefault(); }
      else if (k >= "1" && k <= String(PRESETS.length)) setPreset(PRESETS[+k - 1].id);
    },
  });
  UI.addStopper(function () { clearInterval(st.demoTimer); st.heatToken++; W = null; SAMPLES.length = 0; });
  try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { readColors(); pad.redraw(); render(); }); } catch (e) { /* 旧浏览器 */ }
  st.ready = true;

  // 供自动化验收用（tools/browser_check.py）
  window.LAB_TEST = {
    state: function () {
      return {
        ready: st.ready, mode: st.mode, preset: st.preset, help: UI.helpOpen(), mobile: UI.isMobile(),
        brainOpen: document.body.classList.contains("brain-open"), hasInput: !!st.x, fromUser: st.fromUser,
        pred: st.x ? st.res.pred : null, conf: st.x ? st.res.prob[st.res.pred] : null, base: st.base,
        knocked: maskCount(st.preset), heatReady: !!st.heat, all: st.all,
        tasks: TASKS.map(function (t) { return [t.id, !!taskDone(t.id)]; }), digitsDone: Object.keys(st.tasks.digits).length,
        formula: byId[st.preset].formula, formulaPy: byId[st.preset].structure, params: byId[st.preset].nParams, paramsPy: byId[st.preset].params,
      };
    },
    sample: function (i) { loadSample(i, false); return D.samples.y[st.sample]; },
    sampleLabel: function (i) { return D.samples.y[i]; },
    nSamples: SAMPLES.length,
  };
})();
