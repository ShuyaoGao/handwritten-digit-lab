"""构建产物：file:// 双击能跑（不用 module、不 fetch、不依赖 CDN），署名已填，发布包合规。"""
import re
import sys
import zipfile

import pytest

from common.build_util import credits

GEN = ["lab.js", "core.js", "compress.js", "invert.js"]
ASSETS = ["app.js", "lab.js", "map.js", "main.js", "app.css", "nn.js", "arch.js", "archviz.js", "pad.js", "ui.js", "base.css"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from webapp import build
    out = tmp_path_factory.mktemp("webapp") / "play"
    argv = sys.argv
    sys.argv = ["build", "--out", str(out), "--release"]
    try:
        build.main()
    finally:
        sys.argv = argv
    return out


def test_files(built):
    for f in ["index.html", "双击打开.html"] + [f"assets/{a}" for a in ASSETS if a != "app.js"]:
        assert (built / f).exists(), f
    for g in GEN:
        assert (built / "assets" / "gen" / g).exists(), g


def test_scripts_in_order(built):
    html = (built / "index.html").read_text(encoding="utf-8")
    order = [html.index(f'src="assets/{s}"') for s in ["nn.js", "arch.js", "archviz.js", "pad.js", "ui.js", "gen/lab.js", "lab.js", "map.js", "main.js"]]
    assert order == sorted(order), "脚本顺序：库 → 第一课数据 → 两课 → main.js"
    assert 'src="assets/gen/core.js"' not in html, "第二课的数据要等进入第二课时再加载"


def test_file_protocol_rules(built):
    for f in built.rglob("*"):
        if f.suffix not in (".html", ".js", ".css"):
            continue
        t = f.read_text(encoding="utf-8")
        assert 'type="module"' not in t, f
        assert not re.search(r"\bfetch\s*\(|XMLHttpRequest", t), f
        assert not re.search(r"<script[^>]+src=\"https?:", t), f"{f}：不能从网络加载脚本"
        assert not re.search(r"<link[^>]+href=\"https?:", t), f"{f}：不能从网络加载样式或字体"
        assert "@import" not in t and "serviceWorker" not in t, f
        assert "{{" not in t.replace("{{n}}", ""), f"{f}：还有未替换的 {{KEY}} 占位"


def test_credits_in_footer(built):
    html = (built / "index.html").read_text(encoding="utf-8")
    cr = credits("webapp")
    for k in ["AUTHOR_EN", "UNIVERSITY_EN", "UNIVERSITY_KO", "SERIES_NAME"]:
        assert cr[k] in html
    assert "MIT" in html and "CC BY 4.0" in html and "CC BY-SA 3.0" in html


def test_release_zip(built):
    cr = credits("webapp")
    z = built.parent / "release" / f"{cr['REPO_NAME']}-v{cr['VERSION']}.zip"
    assert z.exists()
    assert z.name.isascii(), "GitHub 会删掉附件文件名里的中文"
    assert z.stat().st_size <= 15 * 1024 * 1024
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert f"{cr['REPO_NAME']}/index.html" in names
        cn = [i for i in zf.infolist() if not i.filename.isascii()]
        assert cn and all(i.flag_bits & 0x800 for i in cn), "中文文件名要带 UTF-8 标记"
        assert all(n.startswith(cr["REPO_NAME"] + "/") for n in names)
