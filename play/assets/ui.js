// 公共界面骨架：课程切换（第一课 / 第二课）+ 每课的模式标签、说明浮层（首次打开自动弹出，H / ? 再打开）、
// 大屏模式、退出、手机菜单、手机竖屏判断（触屏 + 宽度 ≤ 980 + 竖屏 → body.mobile）、横屏提示、「看大脑」全屏。
// 各课用 UI.lesson({...}) 注册，最后 UI.init({...}) 启动。
var UI = (function () {
  "use strict";
  var cfg = null, lessons = [], cur = null, curTab = null, stoppers = [], listeners = [], stopped = false;
  var $ = function (id) { return document.getElementById(id); };

  function on(target, ev, fn, opt) { target.addEventListener(ev, fn, opt); listeners.push([target, ev, fn, opt]); }
  function store(k, v) {
    try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; }
    return null;
  }
  function isTouch() { return (navigator.maxTouchPoints || 0) > 0 || "ontouchstart" in window; }

  function updateDevice() {
    var b = document.body, touch = isTouch(), portrait = innerHeight >= innerWidth;
    b.classList.toggle("touch", touch);
    b.classList.toggle("mobile", touch && innerWidth <= 980 && portrait);
    b.classList.toggle("landscape", touch && !portrait && innerHeight < 560);
    if (!b.classList.contains("mobile")) { b.classList.remove("brain-open"); closeMenu(); }
    if (cur && cur.onLayout) cur.onLayout();
  }

  function openHelp() { $("help").hidden = false; document.body.classList.add("help-open"); var s = $("help").querySelector(".sheet-body"); if (s) s.scrollTop = 0; }
  function closeHelp() { $("help").hidden = true; document.body.classList.remove("help-open"); store(cfg.key + ":help", "1"); }
  function helpOpen() { return !$("help").hidden; }

  function lessonOf(tabId) { return lessons.filter(function (l) { return l.tabs.some(function (t) { return t.id === tabId; }); })[0]; }

  function renderTabs() {
    var tabsEl = document.querySelector(".tabs");
    tabsEl.innerHTML = "";
    tabsEl.setAttribute("role", "tablist");
    cur.tabs.forEach(function (t) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.tab = t.id; b.setAttribute("role", "tab");
      b.innerHTML = "<span class=\"tl\">" + t.name + "</span>" + (t.short ? "<span class=\"ts\">" + t.short + "</span>" : "");
      if (t.title) b.title = t.title;
      b.onclick = function () { setTab(t.id); };
      tabsEl.appendChild(b);
    });
    document.querySelectorAll(".chapters button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.lesson === cur.id); });
  }

  function setTab(id, silent) {
    var L = lessonOf(id);
    if (!L) return;
    var tab = L.tabs.filter(function (t) { return t.id === id; })[0];
    if (L !== cur) {
      if (cur && cur.onLeave) cur.onLeave();
      cur = L;
      document.body.dataset.lesson = L.id;
      renderTabs();
    }
    curTab = id;
    L.lastTab = id;
    document.querySelectorAll(".tabs button").forEach(function (b) { b.setAttribute("aria-selected", b.dataset.tab === id); });
    document.body.classList.toggle("op", !!tab.op);
    document.body.dataset.mode = id;
    document.body.classList.remove("brain-open");
    closeMenu();
    if (!silent && L.onTab) L.onTab(id);
  }

  function openBrain() { document.body.classList.add("brain-open"); if (cur && cur.onBrain) cur.onBrain(true); }
  function closeBrain() { document.body.classList.remove("brain-open"); if (cur && cur.onBrain) cur.onBrain(false); }

  var menuEl = null;
  function closeMenu() { if (menuEl) { menuEl.remove(); menuEl = null; } }
  function toggleMenu() {
    if (menuEl) return closeMenu();
    menuEl = document.createElement("div");
    menuEl.className = "menu";
    menuEl.setAttribute("role", "menu");
    var items = [["操作说明", openHelp]].concat((cur && cur.menuItems) || []).concat([["退出", quit]]);
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = it[0]; b.setAttribute("role", "menuitem");
      b.onclick = function () { closeMenu(); it[1](); };
      menuEl.appendChild(b);
    });
    document.body.appendChild(menuEl);
  }

  function addStopper(fn) { stoppers.push(fn); }

  // 退出：停止主循环、定时器和输入监听，释放数据，尝试关闭标签页
  function quit() {
    if (stopped) return;
    stopped = true;
    lessons.forEach(function (l) { if (l.onLeave) try { l.onLeave(); } catch (e) { /* 忽略 */ } });
    stoppers.forEach(function (fn) { try { fn(); } catch (e) { /* 忽略 */ } });
    listeners.forEach(function (l) { l[0].removeEventListener(l[1], l[2], l[3]); });
    listeners = [];
    try { window.close(); } catch (e) { /* 忽略 */ }
    var d = document.createElement("div");
    d.className = "stopped"; d.id = "stopped";
    d.innerHTML = "<div><p><b>已停止运行。</b></p><p class=\"small\" style=\"margin-top:8px\">可以直接关闭这个标签页；刷新页面可重新开始。</p></div>";
    document.body.innerHTML = "";
    document.body.appendChild(d);
  }

  function lesson(def) { lessons.push(def); }

  function init(c) {
    cfg = c;
    var ch = document.querySelector(".chapters");
    ch.innerHTML = "";
    lessons.forEach(function (l) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.lesson = l.id;
      b.innerHTML = "<span class=\"tl\">" + l.name + "</span><span class=\"ts\">" + (l.short || l.name) + "</span>";
      b.onclick = function () { setTab(l.lastTab || l.tabs[0].id); };
      ch.appendChild(b);
    });
    $("btnHelp").onclick = openHelp;
    $("btnQuit").onclick = quit;
    $("btnBig").onclick = function () {
      var on2 = !document.body.classList.contains("big");
      document.body.classList.toggle("big", on2);
      $("btnBig").setAttribute("aria-pressed", on2);
      store(c.key + ":big", on2 ? "1" : "0");
      if (cur && cur.onLayout) cur.onLayout();
    };
    if (store(c.key + ":big") === "1") { document.body.classList.add("big"); $("btnBig").setAttribute("aria-pressed", "true"); }
    $("btnMenu").onclick = toggleMenu;
    $("helpStart").onclick = closeHelp;
    document.querySelectorAll("[data-open-brain]").forEach(function (b) { b.onclick = openBrain; });
    document.querySelectorAll("[data-close-brain]").forEach(function (b) { b.onclick = closeBrain; });
    on(window, "resize", updateDevice);
    on(window, "orientationchange", updateDevice);
    on(document, "keydown", function (e) {
      if (window.ARCHVIZ && ARCHVIZ.isOpen()) return;
      if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && e.key !== "Escape") return;
      if (e.key === "h" || e.key === "H") { helpOpen() ? closeHelp() : openHelp(); e.preventDefault(); return; }
      if (e.key === "Escape") { if (helpOpen()) closeHelp(); else if (document.body.classList.contains("brain-open")) closeBrain(); closeMenu(); return; }
      if (helpOpen()) return;
      if (cur && cur.onKey) cur.onKey(e);
    });
    on(document, "click", function (e) { if (menuEl && !menuEl.contains(e.target) && e.target !== $("btnMenu")) closeMenu(); }, true);
    updateDevice();
    if (store(c.key + ":help") !== "1") openHelp(); else closeHelp();
    setTab(c.start);
  }

  return { lesson: lesson, init: init, setTab: setTab, openHelp: openHelp, closeHelp: closeHelp, helpOpen: helpOpen, openBrain: openBrain,
    closeBrain: closeBrain, quit: quit, addStopper: addStopper, on: on, isMobile: function () { return document.body.classList.contains("mobile"); },
    stopped: function () { return stopped; }, current: function () { return { lesson: cur && cur.id, tab: curTab }; } };
})();
