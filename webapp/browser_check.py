"""手写数字实验台的浏览器验收（file:// 打开 play/index.html），两课都测。

  python -m webapp.browser_check                 # 默认检查 play/（内部仓库里是 build/webapp/play）
  python -m webapp.browser_check --play play

完整验收用 Chrome：电脑 1920×1080、1366×768；手机（DevTools 模拟，开启触屏）390×844、414×896、360×780（DPR 3），平板 768×1024（DPR 2）。
Edge 只做冒烟测试：无头 Edge 在页面打开约 20 秒后会把它当成后台页，把 requestAnimationFrame 降到每秒约 7 帧
（Edge 的节能机制；实测与输入、焦点模拟、各种关闭节流的启动参数都无关，Chrome 没有这个现象），
依赖动画时长的检查在长会话里会误报。冒烟测试只检查不依赖动画时长的功能。
结果写到 <play 的上级>/release/browser_check.json，截图在 release/browser_check/。
"""
import argparse
import json
import time
from pathlib import Path

from common.build_util import ROOT
from common.cdp import LAYOUT_JS, Browser, Report, installed_browsers, path_in

SEVEN = [(0.22, 0.2), (0.8, 0.2), (0.42, 0.86)]
DESKTOP = [(1920, 1080), (1366, 768)]
MOBILE = [(390, 844, 3), (414, 896, 3), (360, 780, 3), (768, 1024, 2)]
READY = "window.LAB_TEST && LAB_TEST.state().ready"


def lab(b):
    return b.js("LAB_TEST.state()")


def mp(b):
    return b.js("MAP_TEST.state()")


def draw_seven(b, pad):
    b.js("window.scrollTo(0, 0)")
    time.sleep(0.1)
    b.drag(path_in(b.rect(pad), SEVEN))
    time.sleep(0.8)   # DevTools 触屏模拟：一笔结束后约 0.3 秒内的点按会被手势识别吞掉，等状态稳定再操作


def go_center(b, digit):
    b.js("window.scrollTo(0, 0)")
    time.sleep(0.1)
    x, y = b.js(f"MAP_TEST.zToClient(MAP_TEST.centers[{digit}][0], MAP_TEST.centers[{digit}][1])")
    b.drag([(x + 12, y + 12), (x + 6, y + 6), (x, y)])
    time.sleep(0.8)


def to_lesson(b, lesson):
    b.click(f'.chapters button[data-lesson="{lesson}"]')
    if lesson == "shadow":
        return b.wait("window.MAP_TEST && MAP_TEST.state().ready", 20)
    return b.wait("UI.current().lesson === 'rec'", 5)


def arch_checks(b, rep, scope, wrap, info, nlayers, conv_x_frac):
    """结构图：有块；悬停显示神经元数与连接方式；点卷积层播放动画；Esc 关闭"""
    b.js(f"document.querySelector('{wrap}').scrollIntoView({{block:'center'}})")
    time.sleep(0.3)
    r = b.rect(f"{wrap} canvas")
    txt = b.js(f"document.querySelector('{info}').innerText")
    rep.ok(scope, f"结构图：画出 {nlayers} 层，写明参数合计", r and r["w"] > 200 and f"共 {nlayers} 层" in txt and "参数合计" in txt, txt[:80])
    x, y = r["x"] + r["w"] * conv_x_frac, r["y"] + r["h"] * 0.35
    b.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
    time.sleep(0.2)
    txt = b.js(f"document.querySelector('{info}').innerText")
    rep.ok(scope, "悬停卷积层：显示卷积核个数、连接方式、神经元数和参数公式", all(k in txt for k in ["卷积核", "连接方式", "神经元", "参数"]), txt[:120])
    for t in ("mousePressed", "mouseReleased"):
        b.call("Input.dispatchMouseEvent", {"type": t, "x": x, "y": y, "button": "left", "clickCount": 1})
    time.sleep(1.2)
    t1 = b.js("document.querySelector('.am-text').textContent")
    time.sleep(0.6)
    t2 = b.js("document.querySelector('.am-text').textContent")
    rep.ok(scope, "点卷积层：播放\"卷积核滑动\"动画，数值在变", b.js("ARCHVIZ.isOpen()") and "Σ" in t1 and t1 != t2, f"{t1} / {t2}")
    return True


def close_anim(b):
    b.key("Escape", "Escape", 27)
    time.sleep(0.2)
    return not b.js("ARCHVIZ.isOpen()")


def desktop(b, rep, scope, shots):
    s = lab(b)
    rep.ok(scope, "首次打开弹出操作说明（键鼠版）", s["help"] and b.js("getComputedStyle(document.querySelector('.only-desk')).display!=='none'"))
    b.shot(shots / f"{scope}_0_help.png")
    b.key("Escape", "Escape", 27)
    rep.ok(scope, "Esc 关闭说明", not lab(b)["help"])
    b.key("h", "KeyH", 72)
    rep.ok(scope, "H 再次打开说明", lab(b)["help"])
    b.key("h", "KeyH", 72)
    foot = b.js("document.getElementById('credits').innerText")
    rep.ok(scope, "页脚有署名、学校与许可证", all(t in foot for t in ["Shuyao Gao", "aSSIST University", "MIT", "CC BY 4.0", "CC BY-SA 3.0"]), foot)
    s = lab(b)
    rep.ok(scope, "默认第一课 · 演示，打开即有内容", b.js("UI.current().lesson") == "rec" and s["mode"] == "demo" and s["hasInput"] and s["pred"] is not None, s)
    lay = b.js("(function(){var l=document.querySelector('.left.lesson-rec').getBoundingClientRect(), r=document.querySelector('.brain.lesson-rec').getBoundingClientRect(),"
               " p=document.querySelector('#padBox').getBoundingClientRect(); return {lr:l.right, bl:r.left, bw:r.width, lw:l.width, pt:p.top, pb:p.bottom, pw:p.width, ih:innerHeight};})()")
    rep.ok(scope, "左右两栏：左边画板，右边大脑面板", lay["lr"] <= lay["bl"] + 1 and lay["bw"] > lay["lw"], lay)
    rep.ok(scope, "画板完整在首屏", lay["pt"] >= 0 and lay["pb"] <= lay["ih"] and lay["pw"] >= 220, lay)
    L = b.js(LAYOUT_JS)
    rep.ok(scope, "无横向滚动", L["sw"] <= L["iw"] + 1, L)
    b.shot(shots / f"{scope}_1_demo.png")
    good = []
    for pid in ["mlp", "cnn_s", "cnn_m"]:
        b.click(f'.preset[data-id="{pid}"]')
        s = lab(b)
        good.append(s["preset"] == pid and s["formula"] == s["formulaPy"] and s["params"] == s["paramsPy"])
    rep.ok(scope, "三档网络的结构式、参数量与训练端一致", all(good), good)
    w0 = b.js("document.querySelector('.brain.lesson-rec').getBoundingClientRect().width")
    b.click("#btnBig")
    w1 = b.js("document.querySelector('.brain.lesson-rec').getBoundingClientRect().width")
    b.click("#btnBig")
    rep.ok(scope, "大屏模式：右栏变宽", w1 > w0 * 1.1, f"{w0:.0f} → {w1:.0f}")
    b.click('.tabs button[data-tab="draw"]')
    b.click("#btnClear")
    draw_seven(b, "#pad")
    b.wait("LAB_TEST.state().heatReady", 8)
    s = lab(b)
    rep.ok(scope, "鼠标写一个 7 → 判断为 7", s["mode"] == "draw" and s["fromUser"] and s["pred"] == 7, s)
    rep.ok(scope, "热力图（遮挡法）算完", s["heatReady"])
    arch_checks(b, rep, scope, "#archRec", "#archRecInfo", 7, 0.21)
    b.shot(shots / f"{scope}_2_anim_conv.png")
    rep.ok(scope, "Esc 关闭动画", close_anim(b))
    b.js("window.scrollTo(0, 0)")
    b.shot(shots / f"{scope}_3_draw7.png")
    b.click("#c1Tiles .tile")
    k1 = lab(b)["knocked"]
    b.click("#btnKoRand")
    k2 = lab(b)["knocked"]
    b.click("#btnKoReset")
    rep.ok(scope, "点特征图敲掉 1 个；随机敲掉 25%；全部恢复", k1 == 1 and k2 > 10 and lab(b)["knocked"] == 0, f"{k1}, {k2}")
    b.click('.tabs button[data-tab="practice"]')
    s = lab(b)
    rep.ok(scope, "练习模式：任务清单，写过的 7 已打勾", s["mode"] == "practice" and s["digitsDone"] >= 1 and b.js("document.querySelectorAll('#tasks li').length") == 4, s["tasks"])

    # ---------- 第二课 ----------
    rep.ok(scope, "切到第二课：加载模型，演示开始", to_lesson(b, "shadow"))
    z0 = mp(b)["z"]
    time.sleep(1.2)
    s = mp(b)
    rep.ok(scope, "第二课演示：点在地图上自动游走、影子实时更新", s["mode"] == "demo" and s["z"] != z0 and s["shadowPred"] is not None, f"{z0} → {s['z']}")
    rep.ok(scope, "第一课的演示已停止", not b.js("document.querySelector('.lesson-rec').offsetParent") and lab(b)["mode"] == "practice")
    rep.ok(scope, "结构式、参数量与训练端一致（2 维自编码器）", s["formulaOk"])
    lay = b.js("(function(){var l=document.querySelector('.left.lesson-shadow').getBoundingClientRect(), r=document.querySelector('.brain.lesson-shadow').getBoundingClientRect(),"
               " m=document.querySelector('#m-mapBox').getBoundingClientRect(), sh=document.querySelector('#m-shadow').getBoundingClientRect();"
               " return {lr:l.right, bl:r.left, mt:m.top, mw:m.width, sb:sh.bottom, ih:innerHeight};})()")
    rep.ok(scope, "第二课：左边地图、右边大脑；地图和影子都在首屏", lay["lr"] <= lay["bl"] + 1 and lay["mt"] >= 0 and lay["sb"] <= lay["ih"] and lay["mw"] >= 280, lay)
    b.shot(shots / f"{scope}_4_shadow_demo.png")
    b.click('.tabs button[data-tab="map"]')
    go_center(b, 0)
    s = mp(b)
    rep.ok(scope, "鼠标移到地图上「0」的中心 → 影子被认作 0", s["mode"] == "map" and s["shadowPred"] == 0, s["shadowPred"])
    rep.ok(scope, "练习任务：找到影子会打勾", s["digitsFound"] >= 1, s["tasks"])
    draw_seven(b, "#m-pad")
    s = mp(b)
    rep.ok(scope, "在小画板上写 7 → 地图上出现 ★", s["star"] is not None and dict(s["tasks"])["star"], s["star"])
    b.click("#m-btnPlay")
    time.sleep(3.0)
    rep.ok(scope, "播放渐变：从 0 走到 7", dict(mp(b)["tasks"])["interp"])
    b.click("#m-btnGrid")
    time.sleep(0.5)
    b.js("window.scrollTo(0, 0)")
    time.sleep(0.2)
    rep.ok(scope, "铺满影子", mp(b)["mapMode"] == "grid")
    b.shot(shots / f"{scope}_5_map_grid.png")
    b.click("#m-btnPts")
    b.click("#m-btnFull")
    rep.ok(scope, "切到全景", mp(b)["view"] == "full")
    b.click("#m-btnZoom")
    b.click("#m-brainBody .tiles .tile")
    k1 = mp(b)["knocked"]
    b.click("#m-btnKoRand")
    k2 = mp(b)["knocked"]
    b.click("#m-btnKoReset")
    rep.ok(scope, "敲掉解码器通道 1 个；随机敲掉 25%；全部恢复", k1 == 1 and k2 > 10 and mp(b)["knocked"] == 0, f"{k1}, {k2}")
    b.click('.tabs button[data-tab="compress"]')
    rep.ok(scope, "压缩多少：按需加载 4/10/16/32 维模型", b.wait("MAP_TEST.state().loaded.compress && MAP_TEST.state().reconCards===6", 15))
    rep.ok(scope, "结构式、参数量与训练端一致（10 维）", mp(b)["formulaOk"])
    b.click('#m-reconGrid .rc[data-k="32"]')
    rep.ok(scope, "点 32 个数的还原图 → 大脑面板换成 32 维", mp(b)["selD"] == 32 and mp(b)["formulaOk"])
    b.js("(function(){var i=document.getElementById('m-knob0'); i.value=i.max; i.dispatchEvent(new Event('input'));})()")
    rep.ok(scope, "拧旋钮改编码", mp(b)["knob"])
    arch_checks(b, rep, scope, "#m-arch", "#m-archInfo", 12, 0.16)
    rep.ok(scope, "Esc 关闭动画", close_anim(b))
    b.js("window.scrollTo(0, 0)")
    b.shot(shots / f"{scope}_6_compress.png")
    b.click('.tabs button[data-tab="invert"]')
    rep.ok(scope, "识别网络丢了什么：按需加载反演解码器", b.wait("MAP_TEST.state().loaded.invert && MAP_TEST.state().invCards===6 && MAP_TEST.state().galCells===100", 15))
    rep.ok(scope, "结构式、参数量与训练端一致（识别网络）", mp(b)["formulaOk"])
    b.shot(shots / f"{scope}_7_invert.png")
    rep.ok(scope, "切回第一课：状态保留", to_lesson(b, "rec") and lab(b)["mode"] == "practice")
    errs = b.errors()
    rep.ok(scope, "没有 JS 报错", not errs, errs)
    b.click("#btnQuit")
    rep.ok(scope, "退出：停止运行并提示可关闭标签页", b.wait("!!document.getElementById('stopped')", 3))


def mobile(b, rep, scope, shots):
    s = lab(b)
    rep.ok(scope, "识别为手机竖屏布局", s["mobile"])
    rep.ok(scope, "首次打开弹出说明，显示触屏图解", s["help"] and b.js("getComputedStyle(document.querySelector('.only-touch')).display!=='none'"))
    b.shot(shots / f"{scope}_0_help.png")
    b.click("#helpStart")
    rep.ok(scope, "点「开始」关闭说明", not lab(b)["help"])
    L = b.js(LAYOUT_JS)
    rep.ok(scope, "顶栏一行：课程、模式、☰ 都在屏幕内", b.js("(function(){var t=document.querySelector('.top').getBoundingClientRect(); return t.height < 70;})()") and L["sw"] <= L["iw"] + 1, L)
    b.shot(shots / f"{scope}_1_demo.png")
    b.click('.tabs button[data-tab="draw"]')
    time.sleep(0.2)
    L = b.js(LAYOUT_JS)
    rects = {k: b.rect(k) for k in ["#padBox", ".lesson-rec .pad-btns", "#seen", ".top"]}
    inside = all(r and r["y"] >= 0 and r["y"] + r["h"] <= L["ih"] + 1 for r in rects.values())
    rep.ok(scope, "第一课「写」一屏放下：整页不滚动", L["sh"] <= L["ih"] + 1 and L["sw"] <= L["iw"] + 1 and inside, {"layout": L, **rects})
    b.click("#btnClear")
    draw_seven(b, "#pad")
    rep.ok(scope, "手指写一个 7 → 判断为 7", lab(b)["pred"] == 7, lab(b)["pred"])
    b.shot(shots / f"{scope}_2_draw7.png")
    b.click(".lesson-rec .see-brain")
    br = b.rect("#brain")
    arch = b.js("(function(){var w=document.querySelector('#archRec'), c=w.querySelector('canvas'); return c ? {cw: c.getBoundingClientRect().width, ww: w.clientWidth} : null;})()")
    rep.ok(scope, "「看大脑」全屏展开，结构图在里面（可左右滑）", lab(b)["brainOpen"] and br["w"] >= L["iw"] - 1 and arch and arch["cw"] >= arch["ww"], {"brain": br, "arch": arch})
    b.shot(shots / f"{scope}_3_brain.png")
    b.click(".lesson-rec .brain-back button")
    rep.ok(scope, "「返回」关闭大脑面板", not lab(b)["brainOpen"])

    rep.ok(scope, "切到第二课", to_lesson(b, "shadow"))
    b.click('.tabs button[data-tab="map"]')
    time.sleep(0.4)
    L = b.js(LAYOUT_JS)
    rects = {k: b.rect(k) for k in ["#m-mapBox", "#m-shadow", "#m-padBox", "#m-btnPlay", ".top"]}
    inside = all(r and r["y"] >= 0 and r["y"] + r["h"] <= L["ih"] + 1 and r["x"] >= 0 and r["x"] + r["w"] <= L["iw"] + 1 for r in rects.values())
    rep.ok(scope, "第二课「地图」一屏放下：地图、影子、画板、播放都在视口内", L["sh"] <= L["ih"] + 1 and L["sw"] <= L["iw"] + 1 and inside, {"layout": L, **rects})
    go_center(b, 0)
    rep.ok(scope, "手指拖到「0」的中心 → 影子被认作 0", mp(b)["shadowPred"] == 0, mp(b)["shadowPred"])
    draw_seven(b, "#m-pad")
    rep.ok(scope, "手指在小画板写 7 → 地图上出现 ★", mp(b)["star"] is not None)
    b.shot(shots / f"{scope}_4_map.png")
    b.click("#m-slotMap .see-brain")
    br = b.rect("#m-brain")
    rep.ok(scope, "「看大脑」全屏展开（解码器 + 结构图）", mp(b)["brainOpen"] and br["w"] >= L["iw"] - 1 and b.js("!!document.querySelector('#m-arch canvas')"), br)
    b.shot(shots / f"{scope}_5_brain.png")
    b.click(".lesson-shadow .brain-back button")
    rep.ok(scope, "「返回」关闭大脑面板", not mp(b)["brainOpen"])
    b.click('.tabs button[data-tab="compress"]')
    ok = b.wait("MAP_TEST.state().reconCards===6", 15)
    L = b.js(LAYOUT_JS)
    rep.ok(scope, "「压缩」：加载完成，无横向溢出", ok and L["sw"] <= L["iw"] + 1, L)
    b.shot(shots / f"{scope}_6_compress.png")
    b.click('.tabs button[data-tab="invert"]')
    ok = b.wait("MAP_TEST.state().invCards===6 && MAP_TEST.state().galCells===100", 15)
    L = b.js(LAYOUT_JS)
    rep.ok(scope, "「丢了什么」：加载完成，无横向溢出", ok and L["sw"] <= L["iw"] + 1, L)
    b.click("#btnMenu")
    items = b.js("Array.from(document.querySelectorAll('.menu button')).map(function(x){return x.textContent})") or []
    rep.ok(scope, "☰ 菜单里有说明、本课选项、退出", "退出" in items and "操作说明" in items and any("地图" in i for i in items), items)
    b.click("#btnMenu")
    errs = b.errors()
    rep.ok(scope, "没有 JS 报错", not errs, errs)


def edge_smoke(b, rep, scope):
    """不依赖动画时长的检查：打开、说明、两课切换、结构式、写 7、地图取点、结构图说明、无 JS 报错"""
    rep.ok(scope, "首次打开弹出操作说明", lab(b)["help"])
    b.key("Escape", "Escape", 27)
    s = lab(b)
    rep.ok(scope, "第一课演示有内容；结构式与训练端一致", s["hasInput"] and s["formula"] == s["formulaPy"] and s["params"] == s["paramsPy"])
    b.click('.tabs button[data-tab="draw"]')
    b.click("#btnClear")
    draw_seven(b, "#pad")
    rep.ok(scope, "鼠标写一个 7 → 判断为 7", lab(b)["pred"] == 7, lab(b)["pred"])
    txt = b.js("document.querySelector('#archRecInfo').innerText")
    rep.ok(scope, "结构图：7 层、参数合计", "共 7 层" in txt and "参数合计" in txt, txt[:60])
    rep.ok(scope, "切到第二课（加载模型）", to_lesson(b, "shadow"))
    b.click('.tabs button[data-tab="map"]')
    go_center(b, 0)
    rep.ok(scope, "地图上「0」的中心 → 影子被认作 0；结构式一致", mp(b)["shadowPred"] == 0 and mp(b)["formulaOk"], mp(b)["shadowPred"])
    txt = b.js("document.querySelector('#m-archInfo').innerText")
    rep.ok(scope, "第二课结构图：12 层（编码器 + 解码器）", "共 12 层" in txt, txt[:60])
    L = b.js(LAYOUT_JS)
    rep.ok(scope, "无横向滚动", L["sw"] <= L["iw"] + 1, L)
    errs = b.errors()
    rep.ok(scope, "没有 JS 报错", not errs, errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", default=None)
    args = ap.parse_args()
    internal = (ROOT / "tools" / "make_publish.py").exists()
    play = Path(args.play).resolve() if args.play else (ROOT / "build" / "webapp" / "play" if internal else ROOT / "play")
    url = (play / "index.html").as_uri()
    out = play.parent / "release"
    shots = out / "browser_check"
    shots.mkdir(parents=True, exist_ok=True)
    rep = Report()
    browsers = installed_browsers()
    full = browsers.get("Chrome") or next(iter(browsers.values()))
    for W, H in DESKTOP:
        scope = f"{'Chrome' if full == browsers.get('Chrome') else 'Edge'}_{W}x{H}"
        print(f"[{scope}]")
        b = Browser(full, W, H)
        try:
            rep.ok(scope, "打开页面（file://）", b.goto(url, READY))
            desktop(b, rep, scope, shots)
        finally:
            b.close()
    if "Edge" in browsers and browsers["Edge"] != full:
        for W, H in DESKTOP:
            scope = f"Edge_{W}x{H}_smoke"
            print(f"[{scope}]")
            b = Browser(browsers["Edge"], W, H)
            try:
                rep.ok(scope, "打开页面（file://）", b.goto(url, READY))
                edge_smoke(b, rep, scope)
            finally:
                b.close()
    exe = full
    for W, H, dpr in MOBILE:
        scope = f"mobile_{W}x{H}"
        print(f"[{scope}]")
        b = Browser(exe, W, H, dpr, mobile=True)
        try:
            rep.ok(scope, "打开页面（file://）", b.goto(url, READY))
            mobile(b, rep, scope, shots)
        finally:
            b.close()
    n, total = rep.summary()
    (out / "browser_check.json").write_text(json.dumps({"passed": n, "total": total, "browsers": list(browsers), "rows": rep.rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n通过 {n} / {total}；截图在 {shots}")
    raise SystemExit(0 if n == total else 1)


if __name__ == "__main__":
    main()
