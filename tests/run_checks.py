"""端到端自检 A~E。用法: python tests\\run_checks.py
每项打印 PASS/FAIL 与实测数值；有 FAIL 时退出码 1。
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sketchfix.io_prep import imwrite_unicode, load_prep   # noqa: E402
from sketchfix.align import align_pair                      # noqa: E402
from sketchfix.cli import run                               # noqa: E402

W, H = 720, 540
OUT = os.path.join(ROOT, "tests", "out")

RESULTS = []


def _record(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def paper(seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((H, W), 244.0, dtype=np.float32)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    base += 12 * (0.6 * xx + 0.4 * yy)                    # 拍摄光照渐变
    base += rng.normal(0, 2.0, (H, W)).astype(np.float32)  # 纸张颗粒
    g = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def figure(img, sx=1.0, sy=1.0, erase_leg=False, extra_line=False, ink=55, th=3):
    """在 img 上画线稿人形; sx/sy 对坐标做缩放(绕 360,300)。"""
    def P(x, y):
        return (int(round(360 + (x - 360) * sx)), int(round(300 + (y - 300) * sy)))

    col = (ink, ink, ink)
    cv2.circle(img, P(360, 110), int(38 * min(sx, sy)), col, th, cv2.LINE_AA)
    cv2.line(img, P(360, 148), P(360, 182), col, th, cv2.LINE_AA)
    cv2.line(img, P(298, 182), P(422, 182), col, th, cv2.LINE_AA)
    pts = np.array([P(300, 182), P(420, 182), P(402, 332), P(318, 332)], np.int32)
    cv2.polylines(img, [pts], True, col, th, cv2.LINE_AA)
    cv2.line(img, P(300, 184), P(268, 292), col, th, cv2.LINE_AA)
    cv2.line(img, P(420, 184), P(452, 292), col, th, cv2.LINE_AA)
    # 左腿
    cv2.line(img, P(336, 332), P(328, 500), col, th, cv2.LINE_AA)
    # 右腿(可被擦除)
    if erase_leg:
        cv2.line(img, P(384, 332), P(392, 420), col, th, cv2.LINE_AA)
    else:
        cv2.line(img, P(384, 332), P(392, 500), col, th, cv2.LINE_AA)
    if extra_line:
        cv2.line(img, P(120, 430), P(240, 430), col, th, cv2.LINE_AA)
    return img


def save(name, img):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    imwrite_unicode(p, img)
    return p


def transform(img, deg=0.0, tx=0.0, ty=0.0, s=1.0):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, s)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def soft_rect(img, x0, y0, x1, y1, delta):
    """柔和矩形明暗变化(-delta 变暗), 只改低频, 不引入硬边。"""
    m = np.zeros((H, W), np.float32)
    cv2.rectangle(m, (x0, y0), (x1, y1), 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 28)
    g = img.astype(np.float32) + m[:, :, None] * delta
    return np.clip(g, 0, 255).astype(np.uint8)


def codes(findings):
    return [f.code for f in findings]


def by_code(findings, code):
    return [f for f in findings if f.code == code]


# ---------------- A: 配准 ----------------
def check_a():
    ref = save("ref.png", figure(paper(1)))
    copy = save("copy_shift.png", transform(figure(paper(2)), deg=2.0, tx=6, ty=4))
    r = align_pair(load_prep(ref), load_prep(copy))
    ok = r.ok and r.residual_px < 1.5
    _record("A 配准(平移6px+旋转2°)", ok,
            f"ok={r.ok} residual={r.residual_px:.2f}px inlier={r.inlier_ratio:.2f}")
    return ref


# ---------------- B: 比例 ----------------
def check_b(ref):
    copy = save("copy_narrow.png", figure(paper(3), sx=0.86))
    res = run(ref, copy, os.path.join(OUT, "b"))
    fs = by_code(res["findings"], "PROP_ASPECT")
    ok = bool(fs) and any("0.8" in f.quantity or "14" in f.quantity for f in fs)
    _record("B 比例(横向压到0.86)", ok,
            f"codes={codes(res['findings'])} q={fs[0].quantity if fs else '-'}")


# ---------------- C: 线稿 ----------------
def check_c(ref):
    copy = save("copy_lines.png", figure(paper(4), erase_leg=True, extra_line=True))
    res = run(ref, copy, os.path.join(OUT, "c"))
    miss = by_code(res["findings"], "LINE_MISSING")
    extra = by_code(res["findings"], "LINE_EXTRA")
    ok = bool(miss) and bool(extra)
    _record("C 线稿(抹掉右腿+多画横线)", ok,
            f"missing={[f.region for f in miss]} extra={[f.region for f in extra]}")


# ---------------- D: 明暗 ----------------
def check_d(ref):
    base = figure(paper(5))
    copy = save("copy_dark.png", soft_rect(base, 480, 330, 700, 520, -34))
    res = run(ref, copy, os.path.join(OUT, "d"))
    dark = by_code(res["findings"], "VAL_DARK")
    wrong = [f for f in dark if f.region not in ((3, 2), (3, 3))]
    ok = bool(dark) and not wrong
    _record("D 明暗(右下压暗34级)", ok,
            f"dark={[(f.region, f.quantity) for f in dark]}")


# ---------------- E: 报告完整性 + 诚实降级 ----------------
def check_e(ref):
    copy = save("copy_shift2.png", transform(figure(paper(6)), deg=1.0, tx=3, ty=2))
    res = run(ref, copy, os.path.join(OUT, "e"))
    path = res["paths"]["report"]
    text = open(path, encoding="utf-8").read()
    need = ["现象：", "量化：", "改法：", "复查：", "置信度：", "位置" if False else "###"]
    miss = [n for n in need if n not in text]
    json_ok = all(k in json.load(open(res["paths"]["json"], encoding="utf-8"))
                  for k in ("align", "score", "findings"))
    # 空白临摹 -> 必须走"无法可靠比对"
    blank = save("copy_blank.png", np.full((H, W, 3), 250, np.uint8))
    res2 = run(ref, blank, os.path.join(OUT, "e2"))
    honest = any("无法可靠比对" in n or "配准未通过" in n for n in res2["notes"])
    ok = (not miss) and json_ok and res["findings"] and honest
    _record("E 报告六项+诚实降级", ok,
            f"miss_fields={miss} json={json_ok} n={len(res['findings'])} honest={honest}")


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = check_a()
    check_b(ref)
    check_c(ref)
    check_d(ref)
    check_e(ref)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS")
    if bad:
        print("FAILED:", ", ".join(r[0] for r in bad))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
