import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 内部仓库有 tools/make_publish.py；发布仓库没有
INTERNAL = (ROOT / "tools" / "make_publish.py").exists()
