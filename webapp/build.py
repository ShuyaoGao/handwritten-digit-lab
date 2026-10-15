"""构建"手写数字实验台"的网页（play/）和发布包。两课共用一个页面：

  python -m webapp.build                 # 生成 play/（内部仓库里是 build/webapp/play）
  python -m webapp.build --release       # 另外生成 release/handwritten-digit-lab-v1.0.zip

数据来自两次训练的导出结果：
  第一课 results/digit_lab/<run>/export/model_data.json → assets/gen/lab.js（打开页面时加载）
  第二课 results/digit_map/<run>/export/*.json          → assets/gen/core.js、compress.js、invert.js（进入第二课 / 对应模式时才加载）
"""
import argparse
import json
from pathlib import Path

from common.build_util import ROOT, build_play, credits, dir_size, latest_run, make_zip
from common.export import js_assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="play/ 目录")
    ap.add_argument("--release", action="store_true")
    args = ap.parse_args()
    lab = latest_run("digit_lab")
    shadow = latest_run("digit_map")
    data = json.loads((lab / "export" / "model_data.json").read_text(encoding="utf-8"))
    data.pop("check", None)
    gen = {"lab.js": js_assign("DIGIT_LAB_DATA", data)}
    for n in ["core", "compress", "invert"]:
        gen[f"{n}.js"] = js_assign(f"DM_{n.upper()}", json.loads((shadow / "export" / f"{n}.json").read_text(encoding="utf-8")))
    internal = (ROOT / "tools" / "make_publish.py").exists()
    out = Path(args.out) if args.out else (ROOT / "build" / "webapp" / "play" if internal else ROOT / "play")
    build_play("webapp", out, gen)
    print(f"play/ → {out}  {dir_size(out) / 1e6:.2f} MB（第一课 {lab.name}，第二课 {shadow.name}）")
    if args.release:
        cr = credits("webapp")
        z = make_zip(out, out.parent / "release" / f"{cr['REPO_NAME']}-v{cr['VERSION']}.zip", cr["REPO_NAME"])
        print(f"发布包 → {z}  {z.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
