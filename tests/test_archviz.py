"""网络结构图（common/archviz.js）：每一层的形状、神经元数、参数量由实际权重推出，合计与训练端一致；说明文字写对数字。"""
import json

import pytest

from common.build_util import ROOT, latest_run
from common.jsrun import js_context, js_value


@pytest.fixture(scope="module")
def ctx():
    c = js_context(ROOT / "common" / "nn.js", ROOT / "common" / "arch.js", ROOT / "common" / "archviz.js")
    lab = (latest_run("digit_lab") / "export" / "model_data.json").read_text(encoding="utf-8")
    exp = latest_run("digit_map") / "export"
    c.eval("var LAB = " + lab + "; var W = NN.loadWeights(LAB.weights.manifest, LAB.weights.b64);")
    for n in ["core", "compress", "invert"]:
        c.eval(f"var {n.upper()} = " + (exp / f"{n}.json").read_text(encoding="utf-8") + ";")
    c.eval("var W2 = {}; [CORE, COMPRESS, INVERT].forEach(function(o){ NN.loadWeights(o.weights.manifest, o.weights.b64, W2); });")
    return c


def blocks(ctx, expr):
    return js_value(ctx, "(function(){var n=" + expr + "; return n.blocks.map(function(b){return {kind:b.kind, shape:b.shape, params:b.params||0,"
                         " code:!!b.code, out:!!b.isOutput, text:ARCHVIZ.describe(b).join(' ')};});})()")


@pytest.mark.parametrize("pid", ["mlp", "cnn_s", "cnn_m"])
def test_lab_blocks(ctx, pid):
    p = [q for q in json.loads(js_value(ctx, "JSON.stringify(LAB.presets)")) if q["id"] == pid][0]
    bs = blocks(ctx, f"ARCHVIZ.build(W, '{pid}', LAB.presets.filter(function(q){{return q.id==='{pid}'}})[0].layers, [1,28,28], {{outputLast:true}})")
    assert bs[0]["kind"] == "input" and bs[0]["shape"] == [1, 28, 28]
    assert sum(b["params"] for b in bs) == p["params"]
    assert bs[-1]["out"] and bs[-1]["shape"] == [10]
    kinds = [b["kind"] for b in bs]
    assert kinds == (["input", "fc", "fc"] if pid == "mlp" else ["input", "conv", "pool", "conv", "pool", "fc", "fc"])
    for b in bs:
        if b["kind"] == "conv":
            c = b["shape"][0]
            assert f"{c} 个 3×3 卷积核" in b["text"] and "连接方式" in b["text"]
        n = 1
        for s in b["shape"]:
            n *= s
        if b["kind"] != "input":
            assert f"{n:,}" in b["text"] or str(n) in b["text"], b["text"]


@pytest.mark.parametrize("d", ["2", "4", "10", "16", "32"])
def test_autoencoder_hourglass(ctx, d):
    a = json.loads(js_value(ctx, f"JSON.stringify({{params: CORE.ae['{d}'].params}})"))
    enc = blocks(ctx, f"ARCHVIZ.build(W2, 'ae_{d}', CORE.ae['{d}'].encoder, [1,28,28], {{codeLast:true}})")
    dec = blocks(ctx, f"ARCHVIZ.build(W2, 'ae_{d}', CORE.ae['{d}'].decoder, [{d}], {{}})")
    assert sum(b["params"] for b in enc + dec) == a["params"]
    assert enc[-1]["code"] and enc[-1]["shape"] == [int(d)]
    assert [b["kind"] for b in dec] == ["input", "fc", "fc", "reshape", "convT", "convT"]
    assert dec[-1]["shape"] == [1, 28, 28]
    assert "反卷积" in dec[-2]["text"] and "盖章" in dec[-2]["text"]


def test_classifier_blocks(ctx):
    core = json.loads(js_value(ctx, "JSON.stringify({p: CORE.classifier.params})"))
    bs = blocks(ctx, "ARCHVIZ.build(W2, 'classifier', CORE.classifier.layers, [1,28,28], {outputLast:true})")
    assert sum(b["params"] for b in bs) == core["p"]
    # 第二课"识别网络丢了什么"的分支挂在第 2、4、5、6 块（C1、C2、F1、输出层）
    assert [bs[i]["shape"] for i in (2, 4, 5, 6)] == [[16, 14, 14], [32, 7, 7], [128], [10]]
