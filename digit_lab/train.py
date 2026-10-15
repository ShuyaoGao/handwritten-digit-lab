"""手写数字识别实验台：训练三档网络（只有全连接 / 小卷积 / 标准卷积），导出给网页用。

  python -m digit_lab.train                # RTX 5090 Laptop 上三档合计约 20 秒；CPU 也能跑，慢很多
  python -m digit_lab.train --epochs 1     # 冒烟测试

训练时对每个批次做轻微的随机旋转（±12°）、缩放（0.85–1.10）、平移（±2 像素），
让网络对网页上手写的数字更稳；测试集上不做任何增强。

输出（results/digit_lab/<run_id>/）：
  result.json          各档网络的测试准确率、参数量、每类准确率、最易混的数字对
  _env_stamp.json      运行环境
  raw/train_log.jsonl  每个 epoch 的训练损失与准确率
  export/model_data.json  网页用的 float16 权重、层列表、示例图、一致性校验样本
"""
import argparse
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets

from common.export import pack_weights, u8_b64

ROOT = Path(__file__).resolve().parents[1]


# ---------------- 三档网络（层列表与 common/nn.js 一一对应，改这里要同步改 LAYERS） ----------------

class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

    @staticmethod
    def layers():
        return [{"op": "flatten"}, {"op": "linear", "w": "fc1.weight", "b": "fc1.bias"}, {"op": "relu"},
                {"op": "tap", "name": "F1"}, {"op": "linear", "w": "fc2.weight", "b": "fc2.bias"}, {"op": "tap", "name": "out"}]


class CNN(nn.Module):
    def __init__(self, c1, c2, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        h = F.max_pool2d(F.relu(self.conv1(x)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        return self.fc2(F.relu(self.fc1(h.flatten(1))))

    @staticmethod
    def layers():
        return [{"op": "conv", "w": "conv1.weight", "b": "conv1.bias", "pad": 1}, {"op": "relu"}, {"op": "pool"},
                {"op": "tap", "name": "C1"},
                {"op": "conv", "w": "conv2.weight", "b": "conv2.bias", "pad": 1}, {"op": "relu"}, {"op": "pool"},
                {"op": "tap", "name": "C2"},
                {"op": "flatten"}, {"op": "linear", "w": "fc1.weight", "b": "fc1.bias"}, {"op": "relu"},
                {"op": "tap", "name": "F1"}, {"op": "linear", "w": "fc2.weight", "b": "fc2.bias"}, {"op": "tap", "name": "out"}]


PRESETS = [
    {"id": "mlp", "name": "只有全连接", "make": lambda: MLP(128), "layers": MLP.layers()},
    {"id": "cnn_s", "name": "小卷积网络", "make": lambda: CNN(4, 8, 32), "layers": CNN.layers()},
    {"id": "cnn_m", "name": "标准卷积网络", "make": lambda: CNN(16, 32, 128), "layers": CNN.layers()},
]


def structure_text(layers, model):
    """结构式，例如：输入 784（28×28）→ 卷积 16@28×28 → 池化 16@14×14 → … → 输出 10 类（网页按同样规则生成，测试锁定两者一致）"""
    parts, s = ["输入 784（28×28）"], [1, 28, 28]
    for L in layers:
        if L["op"] == "conv":
            w = getattr(model, L["w"].split(".")[0]).weight
            s = [w.shape[0], s[1], s[2]]
            parts.append(f"卷积 {s[0]}@{s[1]}×{s[2]}")
        elif L["op"] == "pool":
            s = [s[0], s[1] // 2, s[2] // 2]
            parts.append(f"池化 {s[0]}@{s[1]}×{s[2]}")
        elif L["op"] == "linear":
            n = getattr(model, L["w"].split(".")[0]).weight.shape[0]
            parts.append(f"输出 {n} 类" if n == 10 else f"全连接 {n}")
    return " → ".join(parts)


# ---------------- 训练 ----------------

def augment(x, gen):
    n = x.size(0)
    ang = (torch.rand(n, generator=gen) * 2 - 1) * math.radians(12)
    sc = 0.85 + torch.rand(n, generator=gen) * 0.25
    tx, ty = [(torch.rand(n, generator=gen) * 2 - 1) * (2 / 14) for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).to(x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


@torch.no_grad()
def predict(model, X, bs=2000):
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / "digit_lab" / run_id
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "export").mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (out / "_env_stamp.json").write_text(json.dumps({
        "run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "args": vars(args) | {"data": "data", "out": None},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    tr = datasets.MNIST(args.data, train=True, download=True)
    te = datasets.MNIST(args.data, train=False, download=True)
    Xtr = (tr.data.float() / 255).unsqueeze(1).to(device)
    Ytr = tr.targets.to(device)
    Xte = (te.data.float() / 255).unsqueeze(1).to(device)
    Yte = te.targets.to(device)
    Yte_np = Yte.cpu().numpy()

    log, models, res = [], {}, {}
    t_all = time.time()
    for k, p in enumerate(PRESETS):
        torch.manual_seed(k)
        model = p["make"]().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        steps = args.epochs * ((len(Xtr) + args.bs - 1) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        gen = torch.Generator().manual_seed(100 + k)
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            model.train()
            tot, cor, n = 0.0, 0, 0
            for idx in torch.randperm(len(Xtr), generator=gen).to(device).split(args.bs):
                x = augment(Xtr[idx], gen)
                logits = model(x)
                loss = F.cross_entropy(logits, Ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item() * len(idx)
                cor += (logits.argmax(1) == Ytr[idx]).sum().item()
                n += len(idx)
            model.eval()
            acc = (predict(model, Xte).argmax(1) == Yte).float().mean().item()
            log.append({"model": p["id"], "epoch": ep, "train_loss": tot / n, "train_acc_aug": cor / n, "test_acc": acc})
            print(f"[{p['id']:>5}] epoch {ep:>2}/{args.epochs}  loss {tot / n:.4f}  test acc {acc:.4f}", flush=True)
        pred = predict(model, Xte).argmax(1).cpu().numpy()
        cm = np.zeros((10, 10), int)
        np.add.at(cm, (Yte_np, pred), 1)
        off = cm + cm.T
        np.fill_diagonal(off, 0)
        pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)[:3]
        res[p["id"]] = {
            "name": p["name"],
            "structure": structure_text(p["layers"], model),
            "params": int(sum(q.numel() for q in model.parameters())),
            "test_acc": {"value": float((pred == Yte_np).mean())},
            "per_class_acc": {str(c): float(cm[c, c] / cm[c].sum()) for c in range(10)},
            "most_confused_pairs": [{"pair": f"{i}-{j}", "errors": n} for n, i, j in pairs],
            "train_seconds": round(time.time() - t0, 1),
        }
        models[p["id"]] = model
    train_seconds = time.time() - t_all

    with open(out / "raw" / "train_log.jsonl", "w", encoding="utf-8") as f:
        for r in log:
            f.write(json.dumps(r) + "\n")

    result = {
        "run_id": run_id,
        "data": {"train_n": len(Xtr), "test_n": len(Xte)},
        "train": {"epochs": args.epochs, "seconds": round(train_seconds, 1),
                  "augment": "每批随机旋转 ±12°、缩放 0.85–1.10、平移 ±2 像素；测试集不增强"},
        "models": res,
        "notes": {"test_acc": "MNIST 测试集 1 万张的准确率"},
    }
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 导出给网页 ----
    # 示例图：测试集每类前 12 张（演示模式轮播、"示例"按钮随机取）
    sidx = np.concatenate([np.where(Yte_np == c)[0][:12] for c in range(10)])
    rng = np.random.default_rng(0)
    rng.shuffle(sidx)
    samples = Xte[sidx, 0].cpu().numpy()
    manifest, b64 = pack_weights({p["id"]: models[p["id"]].state_dict() for p in PRESETS})
    chk = Xte[sidx[:3]]
    check = {"x": chk[:, 0].cpu().numpy().round(6).tolist(), "models": {}}
    with torch.no_grad():
        for p in PRESETS:
            m = models[p["id"]]
            entry = {"logits": m(chk).cpu().numpy().tolist()}
            if p["id"] != "mlp":
                entry["C1"] = F.max_pool2d(F.relu(m.conv1(chk)), 2).cpu().numpy().reshape(3, -1).tolist()
            check["models"][p["id"]] = entry
    data = {
        "run_id": run_id,
        "presets": [{"id": p["id"], "name": p["name"], "layers": p["layers"], "structure": res[p["id"]]["structure"],
                     "params": res[p["id"]]["params"], "test_acc": res[p["id"]]["test_acc"]["value"]} for p in PRESETS],
        "weights": {"manifest": manifest, "b64": b64},
        "samples": {"n": len(sidx), "y": Yte_np[sidx].tolist(), "test_index": sidx.tolist(), "b64": u8_b64(samples)},
        "check": check,
        "env": {"device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch": torch.__version__,
                "epochs": args.epochs, "train_seconds": round(train_seconds, 1)},
    }
    (out / "export" / "model_data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for p in PRESETS:
        r = res[p["id"]]
        print(f"{p['name']}: {r['structure']}  参数 {r['params']:,}  测试准确率 {r['test_acc']['value']:.4f}")
    print(f"输出目录：{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")


if __name__ == "__main__":
    main()
