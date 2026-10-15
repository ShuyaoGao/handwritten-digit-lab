"""用 DevTools 协议驱动本机已装的 Chrome / Edge（无头）做浏览器验收。不下载任何浏览器。

手机尺寸一律用 Emulation.setDeviceMetricsOverride + 触屏模拟，而不是窗口尺寸
（无头 Edge 的窗口最窄约 500px，用窗口尺寸会得到"假溢出"）。
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

BROWSERS = {"Chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "Edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "Chrome (macOS)": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "Chromium (Linux)": "/usr/bin/chromium"}


def installed_browsers():
    return {k: v for k, v in BROWSERS.items() if os.path.exists(v)}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Browser:
    def __init__(self, exe, width=1920, height=1080, dpr=1.0, mobile=False):
        import websocket
        self.port = _free_port()
        self.prof = tempfile.mkdtemp(prefix="cdp_")
        self.proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.prof}",
                                      f"--window-size={max(width, 800)},{max(height, 600)}", "--hide-scrollbars", "--mute-audio",
                                      "--no-first-run", "--disable-gpu", "--allow-file-access-from-files",
                                      # 无头窗口会被当成"后台"而节流 requestAnimationFrame（Edge 尤其明显），验收时关掉节流
                                      "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                                      "--disable-backgrounding-occluded-windows", "--disable-features=CalculateNativeWinOcclusion", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws_url = None
        for _ in range(150):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=1).read())
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.1)
        if not ws_url:
            self.close()
            raise RuntimeError("浏览器调试端口没有响应")
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.mid, self.events = 0, []
        self.call("Runtime.enable")
        self.call("Page.enable")
        # 让页面一直认为自己有焦点、处于活动状态（否则 Edge 可能把无头页面当成后台标签冻结渲染）
        for m, p in [("Emulation.setFocusEmulationEnabled", {"enabled": True}), ("Page.setWebLifecycleState", {"state": "active"})]:
            try:
                self.call(m, p)
            except RuntimeError:
                pass
        self.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": dpr, "mobile": mobile})
        if mobile:
            self.call("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
        self.w, self.h, self.mobile = width, height, mobile

    def call(self, method, params=None):
        self.mid += 1
        mid = self.mid
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def js(self, expr):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in r:
            raise RuntimeError(json.dumps(r["exceptionDetails"], ensure_ascii=False)[:800])
        return r.get("result", {}).get("value")

    def goto(self, url, ready_js, timeout=30):
        self.call("Page.navigate", {"url": url})
        return self.wait(ready_js, timeout)

    def wait(self, expr, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.js(expr):
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def errors(self):
        out = []
        for e in self.events:
            if e.get("method") == "Runtime.exceptionThrown":
                d = e["params"]["exceptionDetails"]
                out.append(d.get("exception", {}).get("description", d.get("text", "exception"))[:300])
            if e.get("method") == "Runtime.consoleAPICalled" and e["params"]["type"] == "error":
                out.append(" ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])[:300])
        return out

    def rect(self, selector):
        return self.js("(function(){var e=document.querySelector(%s); if(!e) return null; var r=e.getBoundingClientRect();"
                       "return {x:r.left, y:r.top, w:r.width, h:r.height, cx:r.left+r.width/2, cy:r.top+r.height/2};})()" % json.dumps(selector))

    def click(self, selector):
        self.js("(function(){var e=document.querySelector(%s); if(e) e.scrollIntoView({block:'center'});})()" % json.dumps(selector))
        r = self.rect(selector)
        if not r:
            raise RuntimeError("找不到元素 " + selector)
        if self.mobile:
            self.touch([(r["cx"], r["cy"])])
        else:
            for t in ("mousePressed", "mouseReleased"):
                self.call("Input.dispatchMouseEvent", {"type": t, "x": r["cx"], "y": r["cy"], "button": "left", "clickCount": 1})
        time.sleep(0.15)

    def key(self, key, code, vk):
        for t in ("keyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", {"type": t, "key": key, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk})
        time.sleep(0.1)

    def drag(self, pts):
        """按一条路径拖动：手机用触摸，电脑用鼠标"""
        if self.mobile:
            return self.touch(pts)
        x, y = pts[0]
        self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        self.call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "buttons": 1, "clickCount": 1})
        for x, y in pts[1:]:
            self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y, "button": "left", "buttons": 1})
            time.sleep(0.008)
        self.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "buttons": 0, "clickCount": 1})
        time.sleep(0.2)

    def touch(self, pts):
        x, y = pts[0]
        self.call("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y, "id": 1}]})
        for x, y in pts[1:]:
            self.call("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y, "id": 1}]})
            time.sleep(0.008)
        self.call("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        time.sleep(0.2)

    def shot(self, path):
        data = base64.b64decode(self.call("Page.captureScreenshot", {"format": "png"})["data"])
        with open(path, "wb") as f:
            f.write(data)

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        shutil.rmtree(self.prof, ignore_errors=True)


def path_in(rect, pts):
    """把 0–1 的相对坐标折线换成元素内的屏幕坐标，并在相邻点之间插值"""
    out = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        for k in range(16):
            t = k / 16
            out.append((rect["x"] + rect["w"] * (x0 + (x1 - x0) * t), rect["y"] + rect["h"] * (y0 + (y1 - y0) * t)))
    out.append((rect["x"] + rect["w"] * pts[-1][0], rect["y"] + rect["h"] * pts[-1][1]))
    return out


LAYOUT_JS = ("(function(){var d=document.documentElement; return {sw:d.scrollWidth, iw:innerWidth, sh:d.scrollHeight, ih:innerHeight};})()")


class Report:
    def __init__(self):
        self.rows = []

    def ok(self, scope, item, cond, detail=""):
        self.rows.append({"scope": scope, "item": item, "pass": bool(cond), "detail": str(detail)[:300]})
        print(f"  {'✓' if cond else '✗'} {item}" + (f"  — {detail}" if detail and not cond else ""), flush=True)
        return cond

    def summary(self):
        n = sum(r["pass"] for r in self.rows)
        return n, len(self.rows)
