"""给第二课的地图补上 1 维和 3 维自编码器（和 2 维同一套结构、同样的训练设置）。

不重训已有模型：读入最新一次 digit_map 运行的 models.pt，用其中冻结的识别网络评价，
新模型 ae_1、ae_3 追加进同一个 models.pt；指标写进 result.json 的 autoencoder_map
（不改 autoencoder 下已有的数字）；测试集编码落 raw/latent_ae1.npz、raw/latent_ae3.npz。
最后重新导出 export/*.json（多出 dims.json）。

用法：python -m digit_map.train_map_dims [--run run_...]     # 约 30 秒（RTX 5090 Laptop）
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

from digit_map.train import AutoEncoder, Classifier, bce_per_img, fit, in_chunks

ROOT = Path(__file__).resolve().parents[1]
MAP_DIMS = [1, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dims", default="1,3")
    ap.add_argument("--data", default=str(ROOT / "data"))
    args = ap.parse_args()
    runs = sorted((ROOT / "results" / "digit_map").glob("run_*"))
    run = ROOT / "results" / "digit_map" / args.run if args.run else runs[-1]
    dims = [int(d) for d in args.dims.split(",")]
    env = json.loads((run / "_env_stamp.json").read_text(encoding="utf-8"))
    targs = env["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stamp = run / "_env_stamp_map_dims.json"
    stamp.write_text(json.dumps({
        "script": "digit_map/train_map_dims.py", "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dims": dims, "ae_epochs": targs["ae_epochs"], "bs": targs["bs"], "seed": 2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    tr = datasets.MNIST(args.data, train=True, download=True)
    te = datasets.MNIST(args.data, train=False, download=True)
    Xtr = (tr.data.float() / 255).unsqueeze(1).to(device)
    Ytr = tr.targets.to(device)
    Xte = (te.data.float() / 255).unsqueeze(1).to(device)
    Yte = te.targets.to(device)

    sds = torch.load(run / "models.pt", map_location="cpu")
    clf = Classifier().to(device)
    clf.load_state_dict(sds["classifier"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)

    log, res = [], {}
    Ytr_np, Yte_np = Ytr.cpu().numpy(), Yte.cpu().numpy()
    t0 = time.time()
    for d in dims:
        torch.manual_seed(2)
        ae = AutoEncoder(d).to(device)
        fit(f"ae_{d}", ae, lambda i, ae=ae: bce_per_img(ae(Xtr[i]), Xtr[i]), Xtr, targs["ae_epochs"], targs["bs"], 2, log)
        ae.eval()
        with torch.no_grad():
            recon = in_chunks(ae, Xte)
            ztr = in_chunks(ae.encode, Xtr).cpu().numpy()
            zte = in_chunks(ae.encode, Xte).cpu().numpy()
            mse_each = ((recon - Xte) ** 2).flatten(1).mean(1)
            ok = in_chunks(clf, recon).argmax(1) == Yte
        r = {
            "dim": d,
            "mse": {"value": mse_each.mean().item()},
            "recog_acc": {"value": ok.float().mean().item()},
            "per_class_mse": {str(c): mse_each[Yte == c].mean().item() for c in range(10)},
            "per_class_recog_acc": {str(c): ok[Yte == c].float().mean().item() for c in range(10)},
        }
        knn = KNeighborsClassifier(n_neighbors=5).fit(ztr, Ytr_np)
        r["code_knn5_acc"] = {"value": float((knn.predict(zte) == Yte_np).mean())}
        sc = StandardScaler().fit(ztr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(ztr), Ytr_np)
        r["code_logreg_acc"] = {"value": float((lr.predict(sc.transform(zte)) == Yte_np).mean())}
        res[str(d)] = r
        np.savez_compressed(run / "raw" / f"latent_ae{d}.npz", z=zte, y=Yte_np)
        sds[f"ae_{d}"] = {n: t.detach().cpu() for n, t in ae.state_dict().items()}
        print(f"ae_{d}: mse {r['mse']['value']:.4f}  recog {r['recog_acc']['value']:.3f}  "
              f"knn {r['code_knn5_acc']['value']:.3f}  logreg {r['code_logreg_acc']['value']:.3f}")

    with open(run / "raw" / "train_log_map_dims.jsonl", "w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    torch.save(sds, run / "models.pt")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["autoencoder_map"] = res
    result["autoencoder_map_train_seconds"] = round(time.time() - t0, 1)
    result["notes"]["autoencoder_map"] = "地图用的 1 维 / 3 维自编码器：结构、轮数、批大小、随机种子与 autoencoder 下的相同，后补训练（digit_map/train_map_dims.py）"
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    from digit_map.export import export
    out = export(run, sds, Xte.cpu(), Yte.cpu(), result)
    for f in sorted(out.glob("*.json")):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
