"""构建公用：读署名配置、替换 {{KEY}}、拷贝前端文件、打 Release 包。"""
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON_ASSETS = ["nn.js", "arch.js", "archviz.js", "pad.js", "ui.js", "base.css"]


def credits(case):
    c = json.loads((ROOT / "credits.json").read_text(encoding="utf-8"))
    out = {k: v for k, v in c.items() if isinstance(v, str) and not k.startswith("_")}
    out.update(c.get("CASES", {}).get(case, {}))
    return out


def fill(text, cr):
    def rep(m):
        return cr.get(m.group(1), m.group(0))
    return re.sub(r"\{\{([A-Z_]+)\}\}", rep, text)


def latest_run(case):
    runs = sorted((ROOT / "results" / case).glob("run_*"))
    if not runs:
        raise SystemExit(f"results/{case}/ 下没有训练结果，先运行 python -m {case}.train")
    return runs[-1]


def build_play(case, out, gen_files):
    """out: play/ 目录；gen_files: {文件名: 内容}，写到 assets/gen/"""
    cr = credits(case)
    web = ROOT / case / "web"
    if out.exists():
        shutil.rmtree(out)
    (out / "assets" / "gen").mkdir(parents=True)
    (out / "index.html").write_text(fill((web / "index.html").read_text(encoding="utf-8"), cr), encoding="utf-8")
    for f in web.iterdir():
        if f.suffix in (".js", ".css") and not f.name.startswith("legacy"):
            (out / "assets" / f.name).write_text(fill(f.read_text(encoding="utf-8"), cr), encoding="utf-8")
    for f in COMMON_ASSETS:
        shutil.copy(ROOT / "common" / f, out / "assets" / f)
    for name, content in gen_files.items():
        (out / "assets" / "gen" / name).write_text(content, encoding="utf-8")
    # 中文名的"双击打开"入口（跳到 index.html），方便不熟悉英文文件名的同学
    (out / "双击打开.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>' + cr["TITLE"] + '</title>'
        '<meta http-equiv="refresh" content="0; url=index.html"><p><a href="index.html">打开' + cr["TITLE"] + '</a></p>\n',
        encoding="utf-8")
    return out


def make_zip(play_dir, zip_path, top):
    """只含 play/ 的发布包；top：zip 内的顶层文件夹名。中文文件名由 zipfile 自动带 UTF-8 标记。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(play_dir.rglob("*")):
            if f.is_file():
                z.write(f, f"{top}/{f.relative_to(play_dir).as_posix()}")
    return zip_path


def dir_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
