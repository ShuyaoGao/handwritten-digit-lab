// 由层列表 + 实际权重形状推出每层输出形状和结构式（不手写数字；tests/ 锁定与 Python 端一致）
var ARCH = (function () {
  "use strict";
  function shapes(W, prefix, layers, inShape) {
    var s = (inShape || [1, 28, 28]).slice(), out = [];
    layers.forEach(function (L) {
      var w = L.w ? W[prefix + "." + L.w].shape : null;
      if (L.op === "conv") s = [w[0], s[1] + 2 * (L.pad || 0) - w[2] + 1, s[2] + 2 * (L.pad || 0) - w[3] + 1];
      else if (L.op === "convT") s = [w[1], (s[1] - 1) * L.stride - 2 * L.pad + w[2], (s[2] - 1) * L.stride - 2 * L.pad + w[3]];
      else if (L.op === "pool") s = [s[0], s[1] >> 1, s[2] >> 1];
      else if (L.op === "flatten") s = [s.reduce(function (a, b) { return a * b; }, 1)];
      else if (L.op === "linear") s = [w[0]];
      else if (L.op === "reshape") s = L.shape.slice();
      out.push({ op: L.op, name: L.name, shape: s.slice() });
    });
    return out;
  }

  // 分类网络的结构式：输入 784（28×28）→ 卷积 16@28×28 → 池化 16@14×14 → … → 全连接 128 → 输出 10 类
  function classifierText(W, prefix, layers) {
    var parts = ["输入 784（28×28）"];
    shapes(W, prefix, layers).forEach(function (r) {
      if (r.op === "conv") parts.push("卷积 " + r.shape[0] + "@" + r.shape[1] + "×" + r.shape[2]);
      else if (r.op === "pool") parts.push("池化 " + r.shape[0] + "@" + r.shape[1] + "×" + r.shape[2]);
      else if (r.op === "linear") parts.push(r.shape[0] === 10 ? "输出 10 类" : "全连接 " + r.shape[0]);
    });
    return parts.join(" → ");
  }

  // 通用结构式（与 digit_map/export.py 的 structure_text 规则相同）：
  // first 开头；linear → "全连接 n"，最后一个 linear 用 lastLinear 模板（如 "编码 {n} 个数"）；
  // reshape → "变形 C@H×W"；convT → "反卷积 C@H×W"；end 结尾（可选）
  function text(W, prefix, layers, inShape, first, lastLinear, end) {
    var parts = [first], rows = shapes(W, prefix, layers, inShape), lastLin = -1;
    rows.forEach(function (r, i) { if (r.op === "linear") lastLin = i; });
    rows.forEach(function (r, i) {
      var s = r.shape;
      if (r.op === "conv") parts.push("卷积 " + s[0] + "@" + s[1] + "×" + s[2]);
      else if (r.op === "convT") parts.push("反卷积 " + s[0] + "@" + s[1] + "×" + s[2]);
      else if (r.op === "pool") parts.push("池化 " + s[0] + "@" + s[1] + "×" + s[2]);
      else if (r.op === "reshape") parts.push("变形 " + s[0] + "@" + s[1] + "×" + s[2]);
      else if (r.op === "linear") parts.push(lastLinear && i === lastLin ? lastLinear.replace("{n}", s[0]) : "全连接 " + s[0]);
    });
    if (end) parts.push(end);
    return parts.join(" → ");
  }

  function tapShapes(W, prefix, layers, inShape) {
    var m = {};
    shapes(W, prefix, layers, inShape).forEach(function (r) { if (r.op === "tap") m[r.name] = r.shape; });
    return m;
  }

  return { shapes: shapes, classifierText: classifierText, text: text, tapShapes: tapShapes };
})();
if (typeof globalThis !== "undefined") globalThis.ARCH = ARCH;
