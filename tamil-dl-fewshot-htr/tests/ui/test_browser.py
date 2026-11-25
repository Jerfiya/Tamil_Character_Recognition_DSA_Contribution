"""Browser tests (Playwright + Chromium) for the web UI; also writes reports/ui_screenshots/*.png.

Run:  python -m pytest tests/ui/test_browser.py -q
The server runs in-process on a free port with a temporary copy of the support set.
Camera tests use Chromium's fake media device fed with a Y4M video of a real handwritten letter.
"""
from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pytest

from .conftest import ROOT, needs_ui_data, one_per_kind, png_bytes, square_png

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = needs_ui_data

SHOTS = ROOT / "reports" / "ui_screenshots"
DEVICES = {
    "laptop": dict(viewport={"width": 1440, "height": 900}),
    "ipad_portrait": dict(viewport={"width": 810, "height": 1080}, has_touch=True, is_mobile=True, device_scale_factor=2),
    "ipad_landscape": dict(viewport={"width": 1080, "height": 810}, has_touch=True, is_mobile=True, device_scale_factor=2),
    "iphone13": dict(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True, device_scale_factor=3),
    "android_small": dict(viewport={"width": 360, "height": 740}, has_touch=True, is_mobile=True, device_scale_factor=2),
}


# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server_url(tmp_path_factory) -> Iterator[str]:
    import uvicorn
    from config import load_config
    from server import create_app
    from service import RecognitionService
    base = tmp_path_factory.mktemp("ui_browser")
    shutil.copytree(ROOT / "data" / "support_set", base / "support_set")
    cfg = load_config()
    cfg["paths"]["support_set"] = str(base / "support_set")
    cfg["ui"]["cache_dir"] = str(base / "cache")
    svc = RecognitionService(cfg)
    svc.load()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=10)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> Dict[str, Path]:
    """Test images on disk + a Y4M video (a real handwritten letter on paper) for the fake camera."""
    import cv2
    d = tmp_path_factory.mktemp("ui_files")
    out = {}
    for kind, s in one_per_kind().items():
        p = d / f"{kind}.png"
        shutil.copyfile(s["path"], p)
        out[kind] = p
        out[f"{kind}_label"] = s["label"]
    (d / "square.png").write_bytes(square_png())
    out["square"] = d / "square.png"
    for i, r in enumerate((60, 75, 90)):
        p = d / f"circle{i}.png"
        p.write_bytes(png_bytes(lambda g, r=r: cv2.circle(g, (128, 128), r, 0, 12)))
        out[f"circle{i}"] = p
    # Y4M: 640x480, static frames of the vowel test image on bright paper
    letter = cv2.imdecode(np.fromfile(str(out["vowel"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    letter = cv2.resize(letter, (210, 210), interpolation=cv2.INTER_CUBIC)
    thr = cv2.threshold(letter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    frame = np.full((480, 640), 232, np.uint8)
    patch = np.where(letter < thr, 40, 232).astype(np.uint8)
    frame[135:345, 215:425] = patch
    y4m = d / "letter.y4m"
    with open(y4m, "wb") as fh:
        fh.write(b"YUV4MPEG2 W640 H480 F10:1 Ip A1:1 C420jpeg\n")
        uv = np.full((240, 320), 128, np.uint8).tobytes()
        for _ in range(10):
            fh.write(b"FRAME\n" + frame.tobytes() + uv + uv)
    out["y4m"] = y4m
    return out


@pytest.fixture(scope="module")
def browser(files):
    with playwright.sync_playwright() as p:
        b = p.chromium.launch(args=[
            "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
            f"--use-file-for-fake-video-capture={files['y4m']}",
            "--host-resolver-rules=MAP lan.test 127.0.0.1",
        ])
        yield b
        b.close()


def new_page(browser, device: str = "laptop", dark: bool = False, wait: bool = True, url: str = "", init: str = "",
             reduced: bool = False):
    ctx = browser.new_context(**DEVICES[device], color_scheme="dark" if dark else "light",
                              reduced_motion="reduce" if reduced else "no-preference")
    if init:
        ctx.add_init_script(init)
    page = ctx.new_page()
    page.errors = []                                            # type: ignore[attr-defined]
    page.on("pageerror", lambda e: page.errors.append(str(e)))  # type: ignore[attr-defined]
    if url:
        page.goto(url)
        if wait:
            ready(page)
    return ctx, page


def ready(page) -> None:
    page.wait_for_function("document.getElementById('splash').hidden === true", timeout=30000)
    page.wait_for_selector(".lcell", state="attached", timeout=20000)
    page.evaluate("document.fonts.ready")


def shot(page, name: str, full: bool = False) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(350)
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=full)


def wait_state(page, state: str, timeout: int = 15000) -> None:
    page.wait_for_function(f"document.getElementById('result-body').dataset.state === '{state}'", timeout=timeout)


def upload(page, path: Path) -> None:
    page.click("#tab-upload")
    page.set_input_files("#file-input", str(path))


def mouse_stroke(page, pts: List[Tuple[float, float]]) -> None:
    box = page.locator("#pad").bounding_box()
    xy = [(box["x"] + x * box["width"], box["y"] + y * box["height"]) for x, y in pts]
    page.mouse.move(*xy[0]); page.mouse.down()
    for x, y in xy[1:]:
        page.mouse.move(x, y, steps=6)
    page.mouse.up()


def touch_stroke(page, cdp, pts: List[Tuple[float, float]], selector: str = "#pad") -> None:
    box = page.locator(selector).bounding_box()
    xy = [(box["x"] + x * box["width"], box["y"] + y * box["height"]) for x, y in pts]
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": xy[0][0], "y": xy[0][1]}]})
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        for t in np.linspace(0, 1, 6)[1:]:
            cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x0 + (x1 - x0) * t, "y": y0 + (y1 - y0) * t}]})
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})


KA = [[(0.18, 0.40), (0.82, 0.40)], [(0.50, 0.18), (0.50, 0.55), (0.40, 0.80), (0.28, 0.72)], [(0.50, 0.55), (0.62, 0.78), (0.78, 0.70)]]


def fake_result(letter: str, conf: float = 0.9) -> Dict:
    info = {"letter": letter, "kind": "vowel", "category_ta": "உயிர்", "category_en": "Vowel", "translit": "x",
            "consonant_row": None, "vowel": letter, "variant_of": None}
    top = [{"letter": letter, "confidence": conf, "translit": "x", "category_ta": "உயிர்", "category_en": "Vowel"}]
    return {"label": letter, "known": True, "score": -1.0, "threshold": -4.4, "confidence": conf, "best": info, "top": top,
            "blank": False, "preprocessed": "data:image/png;base64,", "input_size": 96, "latency_ms": 1.0, "model": "fake"}


# ------------------------------------------------------------------------------------------------
def test_splash_then_ready(browser, server_url):
    ctx, page = new_page(browser, wait=False)
    calls = {"n": 0}

    def health(route):
        calls["n"] += 1
        if calls["n"] < 6:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"ready": False, "status": "loading", "message": "Loading model…"}))
        else:
            route.continue_()
    page.route("**/api/health", health)
    page.goto(server_url)
    page.wait_for_selector("#splash .logo-mark", state="visible")
    assert page.locator("#splash-msg").inner_text() == "Loading model…"
    shot(page, "splash_loading")
    ready(page)
    assert calls["n"] >= 6
    ctx.close()


def test_splash_unreachable_then_retry(browser, server_url):
    ctx, page = new_page(browser, wait=False)
    page.route("**/api/health", lambda r: r.abort())
    page.goto(server_url)
    page.wait_for_selector("#splash-error", state="visible", timeout=16000)
    assert "python ui/run.py" in page.locator("#splash-error").inner_text()
    shot(page, "splash_unreachable")
    page.unroute("**/api/health")
    page.click("#splash-retry")
    ready(page)
    ctx.close()


def test_skeletons_and_slow_prediction(browser, server_url, files):
    ctx, page = new_page(browser, wait=False)
    held: List = []
    page.route("**/api/classes", lambda r: held.append(r))
    page.goto(server_url)
    page.wait_for_function("document.getElementById('splash').hidden === true", timeout=30000)
    assert page.locator(".chart-skeleton").is_visible()
    pred_held: List = []
    page.route("**/api/predict?*", lambda r: pred_held.append(r))
    upload(page, files["vowel"])
    page.wait_for_function("document.getElementById('result-body').dataset.state === 'loading'", timeout=5000)
    page.wait_for_selector(".r-skeleton .sk-msg", timeout=5000)          # "Still working…" after 2 s
    assert page.locator("#predict-btn").is_disabled()
    shot(page, "skeleton_loaders")
    for r in held:
        r.continue_()
    for r in pred_held:
        r.continue_()
    wait_state(page, "result")
    page.wait_for_selector(".lcell", state="attached")
    assert not page.locator("#predict-btn").is_disabled()
    ctx.close()


@pytest.mark.parametrize("dark", [False, True])
def test_laptop_layout(browser, server_url, files, dark):
    ctx, page = new_page(browser, "laptop", dark=dark, url=server_url)
    upload(page, files["compound"])
    wait_state(page, "result")
    page.wait_for_timeout(700)
    cols = page.evaluate("getComputedStyle(document.getElementById('layout')).gridTemplateColumns.split(' ').length")
    assert cols == 3
    assert page.locator(".lcell.hit").count() <= 1
    shot(page, f"laptop_{'dark' if dark else 'light'}")
    assert not page.errors
    ctx.close()


@pytest.mark.parametrize("device", ["ipad_portrait", "ipad_landscape"])
def test_tablet_layout_and_touch(browser, server_url, files, device):
    ctx, page = new_page(browser, device, url=server_url)
    cols = page.evaluate("getComputedStyle(document.getElementById('layout')).gridTemplateColumns.split(' ').length")
    assert cols == 2
    cdp = ctx.new_cdp_session(page)
    for s in KA:
        touch_stroke(page, cdp, s)
        page.wait_for_timeout(120)
    assert page.evaluate("window.__pad.count") == 3
    wait_state(page, "result")
    shot(page, f"tablet_{device.split('_')[1]}", full=device == "ipad_portrait")
    ctx.close()


@pytest.mark.parametrize("device", ["iphone13", "android_small"])
def test_mobile_touch_bottom_sheet(browser, server_url, files, device):
    ctx, page = new_page(browser, device, url=server_url)
    assert page.locator(".bottom-nav").is_visible() and not page.locator(".mode-switch").is_visible()
    assert page.evaluate("getComputedStyle(document.getElementById('pad')).touchAction") == "none"
    cdp = ctx.new_cdp_session(page)
    scroll0 = page.evaluate("window.scrollY")
    for s in KA:
        touch_stroke(page, cdp, s)
        page.wait_for_timeout(150)
    assert page.evaluate("window.scrollY") == scroll0           # drawing never scrolls the page
    wait_state(page, "result")
    page.wait_for_function("document.getElementById('result-card').classList.contains('open')")
    assert page.evaluate("window.__predictCount") == 1
    shot(page, f"mobile_{device}_draw_result")
    # swipe the sheet down by its handle
    touch_stroke(page, cdp, [(0.5, 0.3), (0.5, 3.0), (0.5, 9.0)], selector="#result-handle")
    page.wait_for_function("!document.getElementById('result-card').classList.contains('open')")
    # upload a real sample -> bottom sheet with a confident result
    upload(page, files["vowel"])
    wait_state(page, "result")
    page.wait_for_function("document.getElementById('result-card').classList.contains('open')")
    page.wait_for_timeout(500)
    shot(page, f"mobile_{device}_bottom_sheet")
    page.click("#result-card .sheet-close")
    page.click("#bn-chart")
    page.wait_for_function("document.getElementById('chart-card').classList.contains('open')")
    shot(page, f"mobile_{device}_chart_sheet")
    ctx.close()


def test_no_horizontal_scroll_targets_and_tamil_fonts(browser, server_url, files):
    for w, h, touch in [(320, 640, True), (390, 844, True), (768, 1024, True), (1024, 768, True), (1440, 900, False)]:
        ctx = browser.new_context(viewport={"width": w, "height": h}, has_touch=touch, is_mobile=touch)
        page = ctx.new_page()
        page.goto(server_url)
        ready(page)
        upload(page, files["vowel"])
        wait_state(page, "result")
        sw = page.evaluate("document.documentElement.scrollWidth")
        assert sw <= w, f"horizontal scroll at {w}px: scrollWidth {sw}"
        if touch:
            small = page.evaluate("""() => {
              const bad = [];
              for (const e of document.querySelectorAll('button, select, summary, [role=tab], [role=radio], label.switch, label.check')) {
                const r = e.getBoundingClientRect(); const cs = getComputedStyle(e);
                if (!r.width || !r.height || cs.visibility === 'hidden' || e.closest('[hidden]')) continue;
                if (r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
                if (r.width < 43.5 || r.height < 43.5) bad.push(`${e.tagName}.${e.className}#${e.id}:${r.width.toFixed(0)}x${r.height.toFixed(0)}`);
              }
              return bad; }""")
            assert not small, f"touch targets < 44px at {w}px: {small}"
        fonts = page.evaluate("""async () => { await document.fonts.ready;
            return [document.fonts.check('20px "Noto Sans Tamil"', 'அகௌ'), document.fonts.check('20px "Noto Serif Tamil"', 'அகௌ'),
                    [...document.fonts].filter(f => f.family.includes('Tamil')).map(f => f.status)]; }""")
        assert fonts[0] and fonts[1] and set(fonts[2]) == {"loaded"}, fonts
        ctx.close()


def test_draw_autopredict_once_and_no_stale_overwrite(browser, server_url):
    ctx, page = new_page(browser, "laptop", url=server_url)
    for s in KA:                                            # three strokes, short pauses (< 600 ms)
        mouse_stroke(page, s)
        page.wait_for_timeout(250)
    page.wait_for_timeout(1500)
    assert page.evaluate("window.__predictCount") == 1
    # stale response: first request answers late with அ, second answers at once with க
    held: List = []

    def handler(route):
        if not held:
            held.append(route)
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(fake_result("க")))
    page.route("**/api/predict?*", handler)
    page.click("#predict-btn")                              # request 1: held by the route
    page.wait_for_function("window.__predictCount === 2")
    assert page.locator("#predict-btn").is_disabled()       # busy while in flight
    mouse_stroke(page, [(0.2, 0.2), (0.8, 0.8)])             # keep drawing -> auto-predict = request 2
    page.wait_for_function("document.getElementById('big-letter').textContent === 'க'")
    assert page.evaluate("window.__predictCount") == 3
    try:
        held[0].fulfill(status=200, content_type="application/json", body=json.dumps(fake_result("அ")))
    except Exception:
        pass                                                # the request was aborted by the page: also fine
    page.wait_for_timeout(600)
    assert page.locator("#big-letter").inner_text() == "க"
    # keyboard shortcuts
    page.keyboard.press("c")
    assert page.evaluate("window.__pad.count") == 0
    page.keyboard.press("2"); assert page.get_attribute("#tab-upload", "aria-selected") == "true"
    page.keyboard.press("1"); assert page.get_attribute("#tab-draw", "aria-selected") == "true"
    ctx.close()


def test_camera_snapshot_live_and_stop(browser, server_url):
    ctx, page = new_page(browser, "laptop", url=server_url)
    page.click("#tab-camera")
    page.wait_for_function("document.getElementById('video').videoWidth > 0 && document.getElementById('cam-state').hidden", timeout=15000)
    assert page.evaluate("window.__cam.tracks()") == 1
    shot(page, "camera_overlay")
    page.click("#shutter")
    page.wait_for_selector("#snap-actions", state="visible")
    shot(page, "camera_snapshot_preview")
    page.click("#use-snap")
    wait_state(page, "result")
    shot(page, "camera_snapshot_result")
    page.click("[data-cammode=live]")
    page.wait_for_selector("#live-chip", state="visible")
    page.wait_for_selector("#lc-lock", state="visible", timeout=20000)
    assert page.locator("#use-live").is_enabled()
    shot(page, "camera_live_stable")
    page.click("#use-live")
    wait_state(page, "result")
    page.click("#tab-draw")                                 # leaving the tab stops every track
    page.wait_for_timeout(300)
    assert page.evaluate("window.__cam.tracks()") == 0
    ctx.close()


def test_camera_mobile_fullscreen(browser, server_url):
    ctx, page = new_page(browser, "iphone13", url=server_url)
    page.click("#tab-camera")
    page.wait_for_function("document.getElementById('video').videoWidth > 0 && document.getElementById('cam-state').hidden", timeout=15000)
    box = page.locator("#cam").bounding_box()
    assert box["width"] == 390 and box["height"] >= 800
    shot(page, "camera_mobile_fullscreen")
    page.click("#cam-close")
    assert page.evaluate("window.__cam.tracks()") == 0
    ctx.close()


@pytest.mark.parametrize("err,title,name", [
    ("NotAllowedError", "Camera permission denied", "camera_permission_denied"),
    ("NotFoundError", "No camera found", "camera_not_found"),
    ("NotReadableError", "Camera is busy", "camera_busy"),
])
def test_camera_errors(browser, server_url, err, title, name):
    init = f"""navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('x', '{err}'));"""
    ctx, page = new_page(browser, "laptop", url=server_url, init=init)
    page.click("#tab-camera")
    page.wait_for_selector("#cam-state[data-kind=error]", state="visible")
    assert page.locator("#cam-state-title").inner_text() == title
    assert page.get_by_role("button", name="Upload a photo instead").is_visible()
    shot(page, name)
    page.get_by_role("button", name="Upload a photo instead").click()
    assert page.get_attribute("#tab-upload", "aria-selected") == "true"
    ctx.close()


def test_camera_insecure_context(browser, server_url):
    port = server_url.rsplit(":", 1)[1]
    ctx, page = new_page(browser, "iphone13", url=f"http://lan.test:{port}/")
    page.click("#tab-camera")
    page.wait_for_selector("#cam-state[data-kind=error]", state="visible")
    assert "secure" in page.locator("#cam-state-title").inner_text()
    assert "--lan" in page.locator("#cam-state-msg").inner_text()
    assert page.get_by_role("button", name="Take a photo").is_visible()
    shot(page, "camera_insecure_context")
    ctx.close()


def test_unknown_state(browser, server_url, files):
    ctx, page = new_page(browser, "laptop", url=server_url)
    upload(page, files["square"])
    wait_state(page, "result")
    assert page.locator("#unknown-box").is_visible()
    assert page.locator("#big-letter").inner_text() == "?"
    shot(page, "unknown_rejected")
    page.click("#unknown-enroll")
    assert page.get_attribute("#layout", "data-mode") == "enroll"
    assert page.locator(".en-slot.filled").count() == 1
    ctx.close()


def test_practice(browser, server_url, files):
    ctx, page = new_page(browser, "laptop", url=server_url)
    page.click(".mode-btn[data-mode=practice]")
    target = files["vowel_label"]
    page.click(f".lcell[data-letter='{target}']")
    assert page.locator("#pt-letter").inner_text() == target
    upload(page, files["vowel"])
    page.click("#predict-btn")
    page.wait_for_function("['correct','close','wrong'].includes(document.getElementById('pr-feedback').dataset.state)")
    assert page.get_attribute("#pr-feedback", "data-state") == "correct"
    assert page.locator("#pr-fb-ta").inner_text() == "சரி!"
    assert page.locator("#streak").inner_text() == "1" and page.locator("#acc").inner_text() == "100%"
    shot(page, "practice_correct")
    ctx.close()
    ctx, page = new_page(browser, "iphone13", url=server_url)
    page.click(".bn[data-mode=practice]")
    shot(page, "practice_mobile")
    ctx.close()


def test_enroll(browser, server_url, files):
    ctx, page = new_page(browser, "laptop", url=server_url)
    page.click(".mode-btn[data-mode=enroll]")
    page.click(".lcell[data-letter='தெள']")                       # spelling variant -> warning, must confirm
    assert page.locator("#variant-warn").is_visible()
    upload(page, files["circle0"])
    page.wait_for_function("window.__upload.name === 'circle0.png'")
    page.click("#predict-btn")
    page.wait_for_selector(".en-slot.filled")
    assert page.locator("#en-save").is_disabled()
    page.check("#variant-ok")
    assert page.locator("#en-save").is_enabled()
    shot(page, "enroll_variant_warning")
    page.click(".en-slot button")
    letter = "ளௌ"                                                  # missing from the dataset
    cell = page.locator(f".lcell[data-letter='{letter}']")
    assert "off" in cell.get_attribute("class")
    cell.click()
    for i, k in enumerate(("circle0", "circle1", "circle2")):
        page.set_input_files("#file-input", str(files[k]))
        page.wait_for_function(f"window.__upload.name === '{files[k].name}'")
        page.click("#predict-btn")
        page.wait_for_function(f"document.querySelectorAll('.en-slot.filled').length === {i + 1}")
    assert page.locator(".en-slot.filled").count() == 3
    assert page.locator("#variant-warn").is_visible()              # ளௌ vs ளெள in the dataset
    page.check("#variant-ok")
    shot(page, "enroll_samples")
    page.click("#en-save")
    page.wait_for_selector("#en-progress", state="visible")
    page.wait_for_function(f"!document.querySelector(\".lcell[data-letter='{letter}']\").classList.contains('off')", timeout=20000)
    shot(page, "enroll_saved")
    page.click(".mode-btn[data-mode=recognize]")
    upload(page, files["circle1"])
    wait_state(page, "result")
    assert page.locator("#big-letter").inner_text() == letter         # recognised without retraining
    ctx.close()


def test_model_switch_overlay(browser, server_url):
    ctx, page = new_page(browser, "laptop", url=server_url)
    page.click("#model-badge")
    page.wait_for_selector(".model-list button[aria-current=true]")
    shot(page, "model_picker")
    page.click(".model-list button:has-text('ArcFace · Conv4')")
    page.wait_for_selector("#switch-overlay", state="visible")
    shot(page, "model_switch_overlay")
    page.wait_for_selector("#switch-overlay", state="hidden", timeout=120000)
    page.wait_for_selector(".toast.ok")
    assert "ArcFace" in page.locator("#model-name").inner_text()
    page.request.post(f"{server_url}/api/models/select", data={"id": "mnv3-onnx"})
    ctx.close()


def test_history_drawer(browser, server_url, files):
    ctx, page = new_page(browser, "laptop", url=server_url)
    upload(page, files["vowel"]); wait_state(page, "result")
    upload(page, files["aytham"]); wait_state(page, "result")
    page.wait_for_timeout(400)
    page.click("#history-btn")
    assert page.locator("#history li").count() >= 2
    shot(page, "history_drawer")
    page.locator("#history li button").nth(1).click()
    assert page.locator("#drawer").is_hidden()
    ctx.close()


def test_throttled_network_loading_states(browser, server_url, files):
    """Slow 3G + 4x CPU: splash and skeleton appear, states never flicker back."""
    ctx, page = new_page(browser, "iphone13", wait=False, init="""
        window.__states = [];
        new MutationObserver(() => { const b = document.getElementById('result-body'); if (b) { const s = b.dataset.state;
          if (window.__states[window.__states.length - 1] !== s) window.__states.push(s); } })
          .observe(document, { attributes: true, subtree: true, attributeFilter: ['data-state'] });""")
    cdp = ctx.new_cdp_session(page)
    cdp.send("Network.enable")
    cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 400, "downloadThroughput": 50_000, "uploadThroughput": 50_000})
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": 4})
    page.goto(server_url, timeout=90000, wait_until="commit")
    page.wait_for_selector("#splash .logo-mark", state="visible", timeout=60000)
    shot(page, "throttled_splash")
    ready(page)
    upload(page, files["vowel"])
    wait_state(page, "result", timeout=60000)
    states = page.evaluate("window.__states")
    assert "loading" in states and states[-1] == "result", states
    assert states.index("loading") < states.index("result")
    # a second prediction while a result is shown keeps the old result (no skeleton flash)
    page.evaluate("window.__states = ['result']")
    upload(page, files["aytham"])
    page.wait_for_function("document.getElementById('predict-btn').disabled === false && window.__predictCount >= 2", timeout=60000)
    page.wait_for_timeout(300)
    assert page.evaluate("window.__states") == ["result"]
    ctx.close()


def test_reduced_motion(browser, server_url, files):
    ctx, page = new_page(browser, "laptop", url=server_url, reduced=True)
    dur = page.evaluate("getComputedStyle(document.querySelector('.big-letter')).animationDuration")
    assert dur in ("1e-05s", "0.00001s", "0s") or float(dur.rstrip("s")) < 0.001
    ctx.close()
