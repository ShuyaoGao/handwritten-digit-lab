"""改完 credits.json 后运行：重新生成 play/（页面页脚里的署名、系列名、版本号）。

  python tools/fill_credits.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, "-m", "webapp.build", "--release"], cwd=ROOT, check=True)
print("play/ 已按 credits.json 重建。README、LICENSE、CITATION.cff 里的署名请直接编辑这些文件。")
