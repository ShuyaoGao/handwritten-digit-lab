// 手写板：Pointer Events（鼠标、手指、触控笔），笔画平滑，可撤销；
// toInput() 按 MNIST 的做法预处理：笔迹按长边缩放进 20×20，再按质心放到 28×28 中央，像素值 0–1。
var Pad = (function () {
  "use strict";
  var SIZE = 280;

  function Pad(canvas, opts) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    canvas.width = canvas.height = SIZE;
    this.opts = opts || {};
    this.lineWidth = this.opts.lineWidth || 22;
    this.strokes = [];
    this.cur = null;
    this.pid = null;
    this.sample = null;
    this.pending = false;
    var self = this;
    var pos = function (e) {
      var r = canvas.getBoundingClientRect();
      return [(e.clientX - r.left) * SIZE / r.width, (e.clientY - r.top) * SIZE / r.height];
    };
    this._down = function (e) {
      if (self.pid !== null || self.disabled) return;
      e.preventDefault();
      self.pid = e.pointerId;
      try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* 忽略 */ }
      if (self.sample) { self.sample = null; self.strokes = []; }
      self.cur = [pos(e)];
      self.strokes.push(self.cur);
      self.redraw();
      self.changed(false);
    };
    this._move = function (e) {
      if (e.pointerId !== self.pid || !self.cur) return;
      e.preventDefault();
      var evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      evs.forEach(function (ev) { self.cur.push(pos(ev)); });
      self.redraw();
      self.changed(false);
    };
    this._up = function (e) {
      if (e && e.pointerId !== undefined && e.pointerId !== self.pid) return;
      self.end();
    };
    canvas.addEventListener("pointerdown", this._down);
    canvas.addEventListener("pointermove", this._move);
    canvas.addEventListener("pointerup", this._up);
    canvas.addEventListener("pointercancel", this._up);
    canvas.addEventListener("lostpointercapture", this._up);
    canvas.addEventListener("contextmenu", function (e) { e.preventDefault(); });
    window.addEventListener("blur", function () { self.end(); });
  }

  Pad.prototype.end = function () {
    if (this.pid === null) return;
    this.pid = null;
    this.cur = null;
    this.changed(true);
  };

  // live=false：画的过程中（按帧节流）；live=true：抬笔
  Pad.prototype.changed = function (final) {
    var self = this;
    if (final) { if (this.opts.onEnd) this.opts.onEnd(); return; }
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(function () { self.pending = false; if (self.opts.onDraw) self.opts.onDraw(); });
  };

  Pad.prototype.ink = function () {
    return getComputedStyle(this.cv).getPropertyValue("--pad-ink").trim() || getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#111";
  };

  Pad.prototype.redraw = function () {
    var c = this.ctx;
    c.clearRect(0, 0, SIZE, SIZE);
    if (this.sample) {
      var tmp = document.createElement("canvas");
      tmp.width = tmp.height = 28;
      var t = tmp.getContext("2d"), im = t.createImageData(28, 28), rgb = hexRgb(this.ink());
      for (var i = 0; i < 784; i++) { im.data[4 * i] = rgb[0]; im.data[4 * i + 1] = rgb[1]; im.data[4 * i + 2] = rgb[2]; im.data[4 * i + 3] = Math.round(this.sample[i] * 255); }
      t.putImageData(im, 0, 0);
      c.imageSmoothingEnabled = false;
      c.drawImage(tmp, 0, 0, SIZE, SIZE);
      return;
    }
    c.strokeStyle = c.fillStyle = this.ink();
    c.lineWidth = this.lineWidth;
    c.lineCap = "round";
    c.lineJoin = "round";
    this.strokes.forEach(function (s) {
      if (s.length === 1) { c.beginPath(); c.arc(s[0][0], s[0][1], c.lineWidth / 2, 0, 7); c.fill(); return; }
      // 平滑：经过相邻点中点的二次曲线
      c.beginPath();
      c.moveTo(s[0][0], s[0][1]);
      for (var i = 1; i < s.length - 1; i++) {
        var mx = (s[i][0] + s[i + 1][0]) / 2, my = (s[i][1] + s[i + 1][1]) / 2;
        c.quadraticCurveTo(s[i][0], s[i][1], mx, my);
      }
      c.lineTo(s[s.length - 1][0], s[s.length - 1][1]);
      c.stroke();
    });
  };

  Pad.prototype.clear = function () { this.strokes = []; this.sample = null; this.cur = null; this.pid = null; this.redraw(); };
  Pad.prototype.undo = function () { if (this.sample) { this.clear(); return; } this.strokes.pop(); this.redraw(); };
  Pad.prototype.isEmpty = function () { return !this.sample && this.strokes.length === 0; };
  Pad.prototype.setSample = function (arr) { this.sample = Float32Array.from(arr); this.strokes = []; this.redraw(); };
  Pad.prototype.setLineWidth = function (w) { this.lineWidth = w; this.redraw(); };

  Pad.prototype.toInput = function () {
    if (this.sample) return Float32Array.from(this.sample);
    if (!this.strokes.length) return null;
    var a = this.ctx.getImageData(0, 0, SIZE, SIZE).data, x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
    for (var y = 0; y < SIZE; y++) for (var x = 0; x < SIZE; x++) if (a[4 * (y * SIZE + x) + 3] > 20) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (x1 < 0) return null;
    var bw = x1 - x0 + 1, bh = y1 - y0 + 1, s = 20 / Math.max(bw, bh), cv = this.cv;
    var tmp = document.createElement("canvas");
    tmp.width = tmp.height = 28;
    var t = tmp.getContext("2d", { willReadFrequently: true });
    t.imageSmoothingEnabled = true;
    t.imageSmoothingQuality = "high";
    var place = function (dx, dy) {
      t.clearRect(0, 0, 28, 28);
      t.drawImage(cv, x0, y0, bw, bh, 14 - bw * s / 2 + dx, 14 - bh * s / 2 + dy, bw * s, bh * s);
      return t.getImageData(0, 0, 28, 28).data;
    };
    var d = place(0, 0), m = 0, cx = 0, cy = 0;
    for (var i = 0; i < 784; i++) { var v = d[4 * i + 3]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
    d = place(Math.round(13.5 - cx / m), Math.round(13.5 - cy / m));
    var out = new Float32Array(784);
    for (var j = 0; j < 784; j++) out[j] = d[4 * j + 3] / 255;
    return out;
  };

  function hexRgb(c) {
    c = (c || "").trim();
    if (c[0] === "#") {
      if (c.length === 4) c = "#" + c[1] + c[1] + c[2] + c[2] + c[3] + c[3];
      var n = parseInt(c.slice(1), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var m = c.match(/\d+(\.\d+)?/g);
    return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
  }
  Pad.hexRgb = hexRgb;
  return Pad;
})();

// 把 [0,1] 数组画成 w×h 像素图（前景色 fg、背景色 bg 为 [r,g,b]）
function paintGray(cv, arr, w, h, fg, bg) {
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  var ctx = cv.getContext("2d"), im = ctx.createImageData(w, h);
  for (var i = 0; i < w * h; i++) {
    var t = Math.max(0, Math.min(1, arr[i]));
    im.data[4 * i] = bg[0] + (fg[0] - bg[0]) * t;
    im.data[4 * i + 1] = bg[1] + (fg[1] - bg[1]) * t;
    im.data[4 * i + 2] = bg[2] + (fg[2] - bg[2]) * t;
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
function cssColor(name) { return Pad.hexRgb(getComputedStyle(document.documentElement).getPropertyValue(name)); }
