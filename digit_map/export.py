"""把"数字的影子"的全部模型导出给网页：层列表、float16 权重、编码统计、示例图、一致性校验样本。

  python -m digit_map.export                     # 从最新一次运行的 models.pt 导出
  python -m digit_map.export --run run_2026-10-14_113317

分三个包，网页按模式按需加载（手机上先只下载地图要用的部分）：
  export/core.json      2 维自编码器 + 识别网络 + 1 万张测试图的 2 维编码 + 示例图
  export/compress.json  4 / 10 / 16 / 32 维自编码器
  export/invert.json    识别网络各层的反演解码器 + 40 张固定展示图的各方式还原结果
  export/check.json     校验样本（tests/ 用 V8 跑网页端推理，与 PyTorch 比对）
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.export import pack_weights, u8_b64

ROOT = Path(__file__).resolve().parents[1]
AE_DIMS = [2, 4, 10, 16, 32]


def L(op, **kw):
    return {"op": op, **kw}


def trunk_layers(p=""):
    return [L("conv", w=f"{p}trunk.conv1.weight", b=f"{p}trunk.conv1.bias", pad=1), L("relu"), L("pool"), L("tap", name="C1"),
            L("conv", w=f"{p}trunk.conv2.weight", b=f"{p}trunk.conv2.bias", pad=1), L("relu"), L("pool"), L("tap", name="C2"),
            L("flatten")]


CLASSIFIER = trunk_layers() + [L("linear", w="fc1.weight", b="fc1.bias"), L("relu"), L("tap", name="F1"),
                               L("linear", w="fc2.weight", b="fc2.bias"), L("tap", name="out")]
ENCODER = trunk_layers() + [L("linear", w="fc1.weight", b="fc1.bias"), L("relu"), L("tap", name="H"),
                            L("linear", w="fc2.weight", b="fc2.bias"), L("tap", name="z")]


def vec_decoder(p=""):
    return [L("tap", name="z"), L("linear", w=f"{p}fc1.weight", b=f"{p}fc1.bias"), L("relu"), L("tap", name="D1"),
            L("linear", w=f"{p}fc2.weight", b=f"{p}fc2.bias"), L("relu"), L("reshape", shape=[32, 7, 7]), L("tap", name="D7"),
            L("convT", w=f"{p}up1.weight", b=f"{p}up1.bias", stride=2, pad=1), L("relu"), L("tap", name="D14"),
            L("convT", w=f"{p}up2.weight", b=f"{p}up2.bias", stride=2, pad=1), L("sigmoid"), L("tap", name="out")]


INV_C1 = [L("conv", w="conv.weight", b="conv.bias", pad=1), L("relu"), L("convT", w="up.weight", b="up.bias", stride=2, pad=1), L("sigmoid")]
INV_C2 = [L("conv", w="conv.weight", b="conv.bias", pad=1), L("relu"), L("convT", w="up1.weight", b="up1.bias", stride=2, pad=1), L("relu"),
          L("convT", w="up2.weight", b="up2.bias", stride=2, pad=1), L("sigmoid")]


def structure_text(sd, layers, in_shape, first, last_linear, end=None):
    """结构式。规则与 common/arch.js 的 ARCH.text 相同（tests/ 锁定两者一致）。"""
    s, parts = list(in_shape), [first]
    lin = [i for i, l in enumerate(layers) if l["op"] == "linear"]
    for i, l in enumerate(layers):
        w = sd[l["w"]].shape if "w" in l else None
        if l["op"] == "conv":
            s = [w[0], s[1] + 2 * l["pad"] - w[2] + 1, s[2] + 2 * l["pad"] - w[3] + 1]
            parts.append(f"卷积 {s[0]}@{s[1]}×{s[2]}")
        elif l["op"] == "convT":
            s = [w[1], (s[1] - 1) * l["stride"] - 2 * l["pad"] + w[2], (s[2] - 1) * l["stride"] - 2 * l["pad"] + w[3]]
            parts.append(f"反卷积 {s[0]}@{s[1]}×{s[2]}")
        elif l["op"] == "pool":
            s = [s[0], s[1] // 2, s[2] // 2]
            parts.append(f"池化 {s[0]}@{s[1]}×{s[2]}")
        elif l["op"] == "flatten":
            s = [int(np.prod(s))]
        elif l["op"] == "linear":
            s = [w[0]]
            parts.append(last_linear.format(n=s[0]) if last_linear and i == lin[-1] else f"全连接 {s[0]}")
        elif l["op"] == "reshape":
            s = list(l["shape"])
            parts.append(f"变形 {s[0]}@{s[1]}×{s[2]}")
    if end:
        parts.append(end)
    return " → ".join(parts)


def sd_of(models, name, sub=None):
    sd = models[name].state_dict() if not isinstance(models[name], dict) else models[name]
    return {k: v for k, v in sd.items()}


def export(run, models, Xte, Yte, result):
    """models: {名称: state_dict}；Xte: [N,1,28,28] float tensor（CPU）；Yte: [N] int"""
    import torch.nn.functional as F
    out = run / "export"
    out.mkdir(exist_ok=True)
    gal = np.load(run / "raw" / "gallery.npz")
    lat = np.load(run / "raw" / "latent_ae2.npz")

    def ae_info(d):
        sd = models[f"ae_{d}"]
        enc = {k: v for k, v in sd.items() if not k.startswith("dec.")}
        return {
            "d": d,
            "encoder": ENCODER, "decoder": vec_decoder("dec."),
            "encoder_text": structure_text(enc, ENCODER, [1, 28, 28], "输入 784（28×28）", "编码 {n} 个数"),
            "decoder_text": structure_text(sd, vec_decoder("dec."), [d], f"编码 {d} 个数", None, "输出 784（28×28）"),
            "params": int(sum(v.numel() for v in sd.values())),
        }

    # 各自编码器测试集编码的均值和标准差（"拧旋钮"滑块的范围）
    stats = {}
    with torch.no_grad():
        for d in AE_DIMS:
            z = encode(models[f"ae_{d}"], Xte)
            stats[str(d)] = {"mean": z.mean(0).numpy().round(4).tolist(), "std": z.std(0).numpy().round(4).tolist()}

    z2, y2 = lat["z"], lat["y"]
    centers = [np.median(z2[y2 == c], axis=0).round(4).tolist() for c in range(10)]
    clf_text = structure_text(models["classifier"], CLASSIFIER, [1, 28, 28], "输入 784（28×28）", "输出 {n} 类")
    core = {
        "run_id": result["run_id"],
        "result": result,
        "classifier": {"layers": CLASSIFIER, "text": clf_text, "params": int(sum(v.numel() for v in models["classifier"].values()))},
        "ae": {str(d): ae_info(d) | {"stats": stats[str(d)]} for d in AE_DIMS},
        "weights": dict(zip(["manifest", "b64"], pack_weights({"classifier": models["classifier"], "ae_2": models["ae_2"]}))),
        "latent": {"z": np.round(z2.ravel(), 3).tolist(), "y": y2.tolist(), "centers": centers},
        "gallery": {"y": gal["y"].tolist(), "test_index": gal["idx"].tolist(), "orig": u8_b64(gal["orig"] / 255)},
    }
    compress = {"weights": dict(zip(["manifest", "b64"], pack_weights({f"ae_{d}": models[f"ae_{d}"] for d in AE_DIMS if d != 2})))}
    inv_keys = ["C1", "C2", "F1", "logits"]
    invert = {
        "layers": {"C1": INV_C1, "C2": INV_C2, "F1": vec_decoder(), "logits": vec_decoder()},
        "in_shape": {"C1": [16, 14, 14], "C2": [32, 7, 7], "F1": [128], "logits": [10]},
        "weights": dict(zip(["manifest", "b64"], pack_weights({f"inv_{k}": models[f"inv_{k}"] for k in inv_keys}))),
        "gallery": {k: u8_b64(gal[k] / 255) for k in gal.files if k not in ("idx", "y", "orig")},
    }

    # ---- 校验样本 ----
    from digit_map.train import AutoEncoder, C1Decoder, C2Decoder, Classifier, VecDecoder
    clf = Classifier(); clf.load_state_dict(models["classifier"]); clf.eval()
    chk = Xte[:3]
    check = {"x": chk[:, 0].numpy().round(6).tolist(), "ae": {}, "inv": {}}
    with torch.no_grad():
        fe = clf.features(chk)
        check["classifier"] = {"logits": fe["logits"].tolist(), "C2": fe["C2"].reshape(3, -1).tolist()}
        for d in AE_DIMS:
            ae = AutoEncoder(d); ae.load_state_dict(models[f"ae_{d}"]); ae.eval()
            z = ae.encode(chk)
            h = F.relu(ae.dec.fc2(F.relu(ae.dec.fc1(z))))
            check["ae"][str(d)] = {"z": z.tolist(), "out": ae.dec(z)[:, 0].reshape(3, -1).tolist(), "D7": h.tolist()}
        mods = {"C1": C1Decoder(), "C2": C2Decoder(), "F1": VecDecoder(128), "logits": VecDecoder(10)}
        for k, m in mods.items():
            m.load_state_dict(models[f"inv_{k}"]); m.eval()
            check["inv"][k] = m(fe[k])[:, 0].reshape(3, -1).tolist()

    for name, obj in [("core", core), ("compress", compress), ("invert", invert), ("check", check)]:
        (out / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out


@torch.no_grad()
def encode(sd, X):
    from digit_map.train import AutoEncoder
    d = sd["fc2.weight"].shape[0]
    ae = AutoEncoder(d)
    ae.load_state_dict(sd)
    ae.eval()
    return torch.cat([ae.encode(X[i:i + 2000]) for i in range(0, len(X), 2000)])


def main():
    from torchvision import datasets
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    runs = sorted((ROOT / "results" / "digit_map").glob("run_*"))
    run = ROOT / "results" / "digit_map" / args.run if args.run else runs[-1]
    models = torch.load(run / "models.pt", map_location="cpu")
    te = datasets.MNIST(args.data, train=False, download=True)
    Xte = (te.data.float() / 255).unsqueeze(1)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    out = export(run, models, Xte, te.targets, result)
    for f in sorted(out.glob("*.json")):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
