"""卷积特征还原实验：手写数字识别网络 × 自编码器。

问题：
  A. 识别网络（CNN 分类器）逐层抽取的卷积特征里，还剩多少原图信息？
     做法：冻结分类器，在每一层（C1/C2/F1/logits）的输出上各训一个解码器，看能还原成什么样。
  B. 自编码器直接以"还原"为目标，把图压到 d 维（d = 2/4/10/16/32），能还原得多好？
  C. 两种目标对照：同是 10 个数，分类器的 logits 和 10 维自编码器的编码，各自保留了什么？

评价：
  pixel MSE            重建图与原图的逐像素均方误差（越低越像原图）
  recog_acc            把重建图喂回分类器，被认成原数字的比例（越高越保留"是几"）
  code_knn5 / logreg   只用编码（d 个数）识别数字的准确率（仅自编码器）

输出（results/<run_id>/）：
  result.json          全部指标；键路径可直接作 claims locator，如 inversion.F1.mse
  _env_stamp.json      运行环境
  raw/train_log.jsonl  每个模型每个 epoch 的训练损失
  raw/gallery.npz      固定 40 张测试图 + 各模型重建（uint8）
  raw/latent_ae2.npz   测试集 1 万张的 2 维编码 + 标签
  models.pt            全部模型权重（较大，不进仓库）
  export/*.json        网页用的数据包（见 digit_map/export.py）

用法：python -m digit_map.train            # 约 90 秒（RTX 5090 Laptop）
"""
import argparse
import base64
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import datasets


# ---------------- 网络结构（前端 nn.js 按同样结构逐层复现，改这里要同步改那边） ----------------

class Trunk(nn.Module):
    """卷积抽取：28x28 -> C1 16x14x14 -> C2 32x7x7"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        c1 = F.max_pool2d(F.relu(self.conv1(x)), 2)
        c2 = F.max_pool2d(F.relu(self.conv2(c1)), 2)
        return c1, c2


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        c1, c2 = self.trunk(x)
        f1 = F.relu(self.fc1(c2.flatten(1)))
        return {"C1": c1, "C2": c2, "F1": f1, "logits": self.fc2(f1)}

    def forward(self, x):
        return self.features(x)["logits"]


class VecDecoder(nn.Module):
    """从 d 维向量还原：d -> 128 -> 32x7x7 -> 16x14x14 -> 1x28x28"""
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, 32 * 7 * 7)
        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.fc2(F.relu(self.fc1(z)))).view(-1, 32, 7, 7)
        return torch.sigmoid(self.up2(F.relu(self.up1(h))))


class C2Decoder(nn.Module):
    """从 C2（32x7x7）还原"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

    def forward(self, h):
        return torch.sigmoid(self.up2(F.relu(self.up1(F.relu(self.conv(h))))))


class C1Decoder(nn.Module):
    """从 C1（16x14x14）还原"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.up = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

    def forward(self, h):
        return torch.sigmoid(self.up(F.relu(self.conv(h))))


class AutoEncoder(nn.Module):
    """编码器与分类器同构（同样的卷积抽取），只是训练目标换成"还原原图"。"""
    def __init__(self, d):
        super().__init__()
        self.trunk = Trunk()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, d)
        self.dec = VecDecoder(d)

    def encode(self, x):
        _, c2 = self.trunk(x)
        return self.fc2(F.relu(self.fc1(c2.flatten(1))))

    def forward(self, x):
        return self.dec(self.encode(x))


# ---------------- 训练工具 ----------------

def batches(n, bs, gen, device):
    perm = torch.randperm(n, generator=gen).to(device)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def fit(name, model, step_fn, X, epochs, bs, seed, log):
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(X) + bs - 1) // bs))
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for idx in batches(len(X), bs, gen, X.device):
            loss = step_fn(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        log.append({"model": name, "epoch": ep, "train_loss": tot / n})
        print(f"[{name:>10}] epoch {ep:>2}/{epochs}  loss {tot / n:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def in_chunks(fn, X, bs=2000):
    return torch.cat([fn(X[i:i + bs]) for i in range(0, len(X), bs)])


def bce_per_img(xr, x):
    return F.binary_cross_entropy(xr, x, reduction="sum") / x.size(0)


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf-epochs", type=int, default=10)
    ap.add_argument("--dec-epochs", type=int, default=20)
    ap.add_argument("--ae-epochs", type=int, default=25)
    ap.add_argument("--ae-dims", default="2,4,10,16,32")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ae_dims = [int(d) for d in args.ae_dims.split(",")]

    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / "digit_map" / run_id
    (out / "raw").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)

    (out / "_env_stamp.json").write_text(json.dumps({
        "run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "args": vars(args) | {"data": "data", "out": None},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    tr = datasets.MNIST(args.data, train=True, download=True)
    te = datasets.MNIST(args.data, train=False, download=True)
    Xtr = (tr.data.float() / 255).unsqueeze(1).to(device)
    Ytr = tr.targets.to(device)
    Xte = (te.data.float() / 255).unsqueeze(1).to(device)
    Yte = te.targets.to(device)

    log = []
    t0 = time.time()

    # ---- 1. 识别网络 ----
    torch.manual_seed(0)
    clf = Classifier().to(device)
    fit("classifier", clf, lambda i: F.cross_entropy(clf(Xtr[i]), Ytr[i]), Xtr, args.clf_epochs, args.bs, 0, log)
    for p in clf.parameters():
        p.requires_grad_(False)
    clf_acc = (in_chunks(clf, Xte).argmax(1) == Yte).float().mean().item()
    print(f"classifier test acc = {clf_acc:.4f}")

    # ---- 2. 冻结识别网络，逐层训练"反演解码器" ----
    feat_te = {k: in_chunks(lambda x: clf.features(x)[k], Xte) for k in ["C1", "C2", "F1", "logits"]}
    inv = {"C1": C1Decoder(), "C2": C2Decoder(), "F1": VecDecoder(128), "logits": VecDecoder(10)}
    for k, dec in inv.items():
        torch.manual_seed(1)
        dec.to(device)
        for m in dec.modules():  # 用同一随机种子重新初始化
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        fit(f"inv_{k}", dec, lambda i, k=k, dec=dec: bce_per_img(dec(clf.features(Xtr[i])[k]), Xtr[i]),
            Xtr, args.dec_epochs, args.bs, 1, log)

    # ---- 3. 自编码器 ----
    aes = {}
    for d in ae_dims:
        torch.manual_seed(2)
        ae = AutoEncoder(d).to(device)
        aes[d] = fit(f"ae_{d}", ae, lambda i, ae=ae: bce_per_img(ae(Xtr[i]), Xtr[i]), Xtr, args.ae_epochs, args.bs, 2, log)
    train_seconds = time.time() - t0

    with open(out / "raw" / "train_log.jsonl", "w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")

    # ---- 4. 评价 ----
    def evaluate(recon):
        mse_each = ((recon - Xte) ** 2).flatten(1).mean(1)
        pred = in_chunks(clf, recon).argmax(1)
        ok = pred == Yte
        return {
            "mse": {"value": mse_each.mean().item()},
            "recog_acc": {"value": ok.float().mean().item()},
            "per_class_mse": {str(c): mse_each[Yte == c].mean().item() for c in range(10)},
            "per_class_recog_acc": {str(c): ok[Yte == c].float().mean().item() for c in range(10)},
        }

    recons = {}
    res_inv = {}
    with torch.no_grad():
        for k, dec in inv.items():
            recons[f"inv_{k}"] = in_chunks(dec, feat_te[k])
            res_inv[k] = evaluate(recons[f"inv_{k}"])
            res_inv[k]["dim"] = int(np.prod(feat_te[k].shape[1:]))

    res_ae = {}
    Ytr_np, Yte_np = Ytr.cpu().numpy(), Yte.cpu().numpy()
    for d, ae in aes.items():
        with torch.no_grad():
            recons[f"ae_{d}"] = in_chunks(ae, Xte)
            ztr = in_chunks(ae.encode, Xtr).cpu().numpy()
            zte = in_chunks(ae.encode, Xte).cpu().numpy()
        r = evaluate(recons[f"ae_{d}"])
        r["dim"] = d
        knn = KNeighborsClassifier(n_neighbors=5).fit(ztr, Ytr_np)
        r["code_knn5_acc"] = {"value": float((knn.predict(zte) == Yte_np).mean())}
        sc = StandardScaler().fit(ztr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(ztr), Ytr_np)
        r["code_logreg_acc"] = {"value": float((lr.predict(sc.transform(zte)) == Yte_np).mean())}
        res_ae[str(d)] = r
        if d == 2:
            np.savez_compressed(out / "raw" / "latent_ae2.npz", z=zte, y=Yte_np)
        print(f"ae_{d}: mse {r['mse']['value']:.4f}  recog {r['recog_acc']['value']:.3f}  "
              f"knn {r['code_knn5_acc']['value']:.3f}  logreg {r['code_logreg_acc']['value']:.3f}")

    # 分类器 logits 作为"10 维编码"时的同口径探针（与 ae_10 对照）
    with torch.no_grad():
        ltr = in_chunks(clf, Xtr).cpu().numpy()
    lte = feat_te["logits"].cpu().numpy()
    knn = KNeighborsClassifier(n_neighbors=5).fit(ltr, Ytr_np)
    res_inv["logits"]["code_knn5_acc"] = {"value": float((knn.predict(lte) == Yte_np).mean())}

    # ---- 5. 固定展示图：每类前 4 张 ----
    gidx = np.concatenate([np.where(Yte_np == c)[0][:4] for c in range(10)])
    to_u8 = lambda t: (t[gidx, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
    gallery = {"idx": gidx, "y": Yte_np[gidx], "orig": to_u8(Xte)}
    gallery.update({k: to_u8(v) for k, v in recons.items()})
    np.savez_compressed(out / "raw" / "gallery.npz", **gallery)

    result = {
        "run_id": run_id,
        "data": {"train_n": len(Xtr), "test_n": len(Xte)},
        "train_seconds": round(train_seconds, 1),
        "classifier": {"test_acc": {"value": clf_acc}},
        "inversion": res_inv,
        "autoencoder": res_ae,
        "notes": {
            "mse": "测试集 1 万张，逐像素均方误差，像素值 0–1",
            "recog_acc": "把重建图喂回同一个分类器，被认成原数字的比例；原图为 classifier.test_acc",
            "code_knn5_acc": "在训练集编码上拟合 5-NN，测试集准确率；随机猜 0.10",
        },
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 6. 导出：PyTorch 权重 + 网页用的数据包 ----
    models = {"classifier": clf, **{f"inv_{k}": v for k, v in inv.items()}, **{f"ae_{d}": m for d, m in aes.items()}}
    sds = {k: {n: t.detach().cpu() for n, t in m.state_dict().items()} for k, m in models.items()}
    torch.save(sds, out / "models.pt")
    from digit_map.export import export
    export(out, sds, Xte.cpu(), Yte.cpu(), result)

    print(json.dumps({k: v for k, v in result.items() if k != "notes"}, ensure_ascii=False, indent=1)[:3000])
    print(f"输出目录：results/digit_map/{run_id}")


if __name__ == "__main__":
    main()
