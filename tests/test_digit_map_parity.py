"""数字的影子：网页端推理（common/nn.js）与 PyTorch 一致；结构式、参数量与训练端一致。"""
import json

import numpy as np
import pytest

from common.build_util import ROOT, latest_run
from common.jsrun import js_context, js_value


@pytest.fixture(scope="module")
def dm():
    exp = latest_run("digit_map") / "export"
    ctx = js_context(ROOT / "common" / "nn.js", ROOT / "common" / "arch.js")
    for n in ["core", "compress", "invert", "dims"]:
        ctx.eval(f"var {n.upper()} = " + (exp / f"{n}.json").read_text(encoding="utf-8") + ";")
    ctx.eval("var W = {}; [CORE, COMPRESS, INVERT, DIMS].forEach(function(o){ NN.loadWeights(o.weights.manifest, o.weights.b64, W); });")
    check = json.loads((exp / "check.json").read_text(encoding="utf-8"))
    return ctx, check, json.loads((exp / "core.json").read_text(encoding="utf-8"))


def run_all(ctx, x):
    ctx.eval("var x = new Float32Array(%s);" % json.dumps(np.ravel(x).tolist()))
    return js_value(ctx, """(function(){
      var c = NN.run(W, 'classifier', CORE.classifier.layers, x), o = {logits: Array.from(c.out), C2: Array.from(c.taps.C2.d), ae: {}, inv: {}};
      var AEALL = Object.assign({}, CORE.ae, DIMS.ae); Object.keys(AEALL).forEach(function(d){ var a = AEALL[d], z = NN.run(W, 'ae_'+d, a.encoder, x).out;
        var r = NN.run(W, 'ae_'+d, a.decoder, z, {inShape: [+d]}); o.ae[d] = {z: Array.from(z), out: Array.from(r.out), D7: Array.from(r.taps.D7.d)}; });
      ['C1','C2','F1','logits'].forEach(function(k){ var t = k === 'logits' ? c.out : c.taps[k].d;
        o.inv[k] = Array.from(NN.run(W, 'inv_'+k, INVERT.layers[k], t, {inShape: INVERT.in_shape[k]}).out); });
      return o; })()""")


def test_forward_matches_pytorch(dm):
    ctx, check, _ = dm
    for i, x in enumerate(check["x"]):
        r = run_all(ctx, x)
        assert np.abs(np.array(r["logits"]) - np.array(check["classifier"]["logits"][i])).max() < 0.02
        assert np.abs(np.array(r["C2"]) - np.array(check["classifier"]["C2"][i])).max() < 0.01
        for d, ref in check["ae"].items():
            assert np.abs(np.array(r["ae"][d]["z"]) - np.array(ref["z"][i])).max() < 0.03, d
            assert np.abs(np.array(r["ae"][d]["out"]) - np.array(ref["out"][i])).max() < 0.01, d   # 像素 0–1
            assert np.abs(np.array(r["ae"][d]["D7"]) - np.array(ref["D7"][i])).max() < 0.05, d
        for k, ref in check["inv"].items():
            assert np.abs(np.array(r["inv"][k]) - np.array(ref[i])).max() < 0.01, k


def test_structure_and_params_locked(dm):
    ctx, _, core = dm
    assert js_value(ctx, "ARCH.text(W,'classifier',CORE.classifier.layers,[1,28,28],'输入 784（28×28）','输出 {n} 类')") == core["classifier"]["text"]
    assert js_value(ctx, "NN.paramCount(W,'classifier',CORE.classifier.layers)") == core["classifier"]["params"]
    for d, a in core["ae"].items():
        assert js_value(ctx, f"ARCH.text(W,'ae_{d}',CORE.ae['{d}'].encoder,[1,28,28],'输入 784（28×28）','编码 {{n}} 个数')") == a["encoder_text"]
        assert js_value(ctx, f"ARCH.text(W,'ae_{d}',CORE.ae['{d}'].decoder,[{d}],'编码 {d} 个数',null,'输出 784（28×28）')") == a["decoder_text"]
        assert js_value(ctx, f"NN.paramCount(W,'ae_{d}',CORE.ae['{d}'].encoder)+NN.paramCount(W,'ae_{d}',CORE.ae['{d}'].decoder)") == a["params"]


def test_latent_matches_raw(dm):
    _, _, core = dm
    lat = np.load(latest_run("digit_map") / "raw" / "latent_ae2.npz")
    assert len(core["latent"]["y"]) == len(lat["y"]) == 10000
    assert np.abs(np.array(core["latent"]["z"]) - lat["z"].ravel()).max() < 1e-3


def test_map_dims_1_and_3(dm):
    """地图的 1 维 / 3 维：结构式、参数量、编码与原始数据一致；和 2 维是同一套结构，只有瓶颈宽度不同。"""
    ctx, _, core = dm
    dims = json.loads((latest_run("digit_map") / "export" / "dims.json").read_text(encoding="utf-8"))
    assert sorted(dims["ae"]) == ["1", "3"]
    for d, a in dims["ae"].items():
        assert js_value(ctx, f"ARCH.text(W,'ae_{d}',DIMS.ae['{d}'].decoder,[{d}],'编码 {d} 个数',null,'输出 784（28×28）')") == a["decoder_text"]
        assert js_value(ctx, f"NN.paramCount(W,'ae_{d}',DIMS.ae['{d}'].encoder)+NN.paramCount(W,'ae_{d}',DIMS.ae['{d}'].decoder)") == a["params"]
        assert a["encoder"] == core["ae"]["2"]["encoder"] and a["decoder"] == core["ae"]["2"]["decoder"]
        lat = np.load(latest_run("digit_map") / "raw" / f"latent_ae{d}.npz")
        assert (lat["y"] == np.array(core["latent"]["y"])).all()
        assert np.abs(np.array(dims["latent"][d]["z"]) - lat["z"].ravel()).max() < 1e-3
        assert len(dims["latent"][d]["centers"]) == 10 and len(dims["latent"][d]["centers"][0]) == int(d)
