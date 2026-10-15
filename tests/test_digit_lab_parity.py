"""手写数字识别实验台：网页端推理（common/nn.js）与 PyTorch 一致；结构式、参数量与训练端一致。"""
import json

import numpy as np
import pytest

from common.build_util import ROOT, latest_run
from common.jsrun import js_context, js_value


@pytest.fixture(scope="module")
def lab():
    data = json.loads((latest_run("digit_lab") / "export" / "model_data.json").read_text(encoding="utf-8"))
    ctx = js_context(ROOT / "common" / "nn.js", ROOT / "common" / "arch.js")
    ctx.eval("var D = " + json.dumps(data) + "; var W = NN.loadWeights(D.weights.manifest, D.weights.b64);")
    return data, ctx


def test_three_presets(lab):
    data, _ = lab
    assert [p["id"] for p in data["presets"]] == ["mlp", "cnn_s", "cnn_m"]


@pytest.mark.parametrize("pid", ["mlp", "cnn_s", "cnn_m"])
def test_forward_matches_pytorch(lab, pid):
    data, ctx = lab
    ref = data["check"]["models"][pid]
    for i, x in enumerate(data["check"]["x"]):
        r = js_value(ctx, "(function(){var p=D.presets.filter(function(q){return q.id===%s})[0];"
                          "var r=NN.run(W,p.id,p.layers,new Float32Array(%s));"
                          "return {out:Array.from(r.out), C1:r.taps.C1?Array.from(r.taps.C1.d):null};})()" % (json.dumps(pid), json.dumps(np.ravel(x).tolist())))
        # 权重是 float16，logit 允许千分之几的误差；类别必须一致
        assert np.abs(np.array(r["out"]) - np.array(ref["logits"][i])).max() < 0.02
        assert int(np.argmax(r["out"])) == int(np.argmax(ref["logits"][i]))
        if "C1" in ref:
            assert np.abs(np.array(r["C1"]) - np.array(ref["C1"][i])).max() < 0.01


@pytest.mark.parametrize("pid", ["mlp", "cnn_s", "cnn_m"])
def test_structure_and_params_locked(lab, pid):
    data, ctx = lab
    p = [q for q in data["presets"] if q["id"] == pid][0]
    sel = "D.presets.filter(function(q){return q.id===%s})[0]" % json.dumps(pid)
    assert js_value(ctx, f"ARCH.classifierText(W, {json.dumps(pid)}, {sel}.layers)") == p["structure"]
    assert js_value(ctx, f"NN.paramCount(W, {json.dumps(pid)}, {sel}.layers)") == p["params"]


def test_result_numbers_match_export(lab):
    data, _ = lab
    res = json.loads((latest_run("digit_lab") / "result.json").read_text(encoding="utf-8"))
    for p in data["presets"]:
        assert res["models"][p["id"]]["test_acc"]["value"] == p["test_acc"]
        assert res["models"][p["id"]]["params"] == p["params"]
