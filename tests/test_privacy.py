"""隐私扫描：要公开的文件里不能出现主机名、本机绝对路径、内网 IP、内部服务器说明、密钥形态的字符串。

内部仓库（同时含两个案例）里只扫描构建出的网页；发布仓库里扫描全部文件。
"""
import re

from common.build_util import ROOT
from tests.conftest import INTERNAL

PATTERNS = {
    "主机名": r"LAPTOP-[A-Z0-9]{6,}|DESKTOP-[A-Z0-9]{6,}|gpu-server|NAS218PLAY",
    "Windows 用户目录": r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+",
    "类 Unix 用户目录": r"/(?:home|Users)/[a-z][a-z0-9_-]+/",
    "内网 / Tailscale IP": r"\b(?:192\.168|10\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}\b",
    "内部命令与目录": r"\btsq0[1s]?\b|\binfer-py\b|labsync@|/volume1/|\bssh gpu\b",
    "密钥形态": r"gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
TEXT = {".py", ".js", ".html", ".css", ".md", ".json", ".jsonl", ".txt", ".cff", ".ini", ".log", ".toml", ".yaml", ".yml", ""}


def files_to_scan():
    if INTERNAL:   # 内部仓库：只扫描构建出的网页
        return [f for f in (ROOT / "build").rglob("*") if f.is_file() and "browser_check" not in f.parts]
    skip = {".git", "__pycache__", ".pytest_cache", "release", "data", ".venv"}
    return [f for f in ROOT.rglob("*") if f.is_file() and not (set(f.relative_to(ROOT).parts) & skip)
            and not f.name.startswith("_scrub_report") and f.name != "test_privacy.py"]


def test_no_private_strings():
    hits = []
    for f in files_to_scan():
        if f.suffix.lower() not in TEXT:
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        for name, pat in PATTERNS.items():
            for m in re.finditer(pat, t):
                hits.append(f"{f.relative_to(ROOT)}: {name}: {m.group(0)[:60]}")
    assert not hits, "\n".join(hits[:40])
