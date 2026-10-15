"""在 V8（py_mini_racer）里运行前端 JS，供测试比对浏览器端推理与 PyTorch。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ATOB = ("var atob = function (s) { var t='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/', o=[], b=0, n=0;"
        " for (var i=0;i<s.length;i++) { var ch=s[i]; if (ch==='=') break; b=((b<<6)|t.indexOf(ch))&0xffffff; n+=6;"
        " if (n>=8) { n-=8; o.push(String.fromCharCode((b>>n)&255)); } } return o.join(''); };")


def js_context(*files):
    from py_mini_racer import MiniRacer
    ctx = MiniRacer()
    ctx.eval("var window = globalThis;" + ATOB)
    for f in files:
        ctx.eval(Path(f).read_text(encoding="utf-8"))
    return ctx


def js_value(ctx, expr):
    return json.loads(ctx.eval(f"JSON.stringify({expr})"))
