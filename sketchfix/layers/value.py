"""明暗关系层：在低频图上逐格比较样张与临摹的黑白灰。

只依赖 ``sketchfix.types``。核心思路：先把笔触高频糊掉，只留下明暗大块，
再按 4x4 网格比较每格的均值与标准差，最后把相邻同类格合并成一条建议。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..types import (GRID_COLS, GRID_ROWS, Finding, cell_label, clamp01,
                     grid_cell)

# ---- 判定阈值（8-bit 灰度级） -----------------------------------------
DARK_DELTA = 10.0        # copy_mean - ref_mean < -10：画脏/压暗
BRIGHT_DELTA = 10.0      # copy_mean - ref_mean > +10：该亮没亮起来
CONTRAST_RATIO = 0.65    # copy_std < ref_std * 0.65：对比低 35% 以上
CONTRAST_MIN_STD = 8.0   # 样张该格本身太平时不判“发灰”
MIN_VALID_COVERAGE = 0.5  # 有效像素不足 50% 的格直接跳过
MAX_PER_CODE = 2         # 每类最多 2 条
MAX_FINDINGS = 4         # 总共最多 4 条


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _norm_gray(img: Any) -> np.ndarray:
    """统一成 uint8 灰度；支持 bool / float(0~1 或 0~255) / BGR。"""
    if img is None:
        return np.zeros((0, 0), np.uint8)
    a = np.asarray(img)
    if a.size == 0:
        return np.zeros((0, 0), np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        if float(a.max()) <= 1.0:
            a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def lowpass(gray: np.ndarray) -> np.ndarray:
    """大核高斯：去掉笔触高频，只留明暗块。"""
    g = _norm_gray(gray)
    if g.size == 0:
        return g
    h, w = g.shape[:2]
    sigma = max(2.0, max(h, w) * 0.006)
    return cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def _gauss_win(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    k = cv2.getGaussianKernel(size, sigma)
    return np.outer(k, k)


def _win_filter(z: np.ndarray, win: np.ndarray) -> np.ndarray:
    """11x11 窗滤波；小图先复制边缘补边，避免 filter2D 越界。"""
    ph, pw = win.shape[0] // 2, win.shape[1] // 2
    if z.shape[0] < win.shape[0] or z.shape[1] < win.shape[1]:
        zp = cv2.copyMakeBorder(z, ph, ph, pw, pw, cv2.BORDER_REPLICATE)
        out = cv2.filter2D(zp, -1, win, borderType=cv2.BORDER_CONSTANT)
        return out[ph:ph + z.shape[0], pw:pw + z.shape[1]]
    return cv2.filter2D(z, -1, win, borderType=cv2.BORDER_REFLECT_101)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """结构相似度：高斯窗 11x11 / sigma 1.5，纯 numpy 公式实现。"""
    x = _norm_gray(a).astype(np.float64)
    y = _norm_gray(b).astype(np.float64)
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape != y.shape:
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_AREA)

    win = _gauss_win(11, 1.5)
    mu1 = _win_filter(x, win)
    mu2 = _win_filter(y, win)
    mu1_sq, mu2_sq = mu1 * mu1, mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = np.maximum(_win_filter(x * x, win) - mu1_sq, 0.0)
    sigma2_sq = np.maximum(_win_filter(y * y, win) - mu2_sq, 0.0)
    sigma12 = _win_filter(x * y, win) - mu12

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


# ---------------------------------------------------------------------------
# 网格统计
# ---------------------------------------------------------------------------
def _grid_stats(low: np.ndarray, valid: np.ndarray,
                rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐格算有效像素均值 / 标准差 / 有效率。无效格为 nan。"""
    h, w = low.shape[:2]
    means = np.full((rows, cols), np.nan, np.float64)
    stds = np.full((rows, cols), np.nan, np.float64)
    cover = np.zeros((rows, cols), np.float64)
    if h <= 0 or w <= 0:
        return means, stds, cover

    # 与 line/regions 层保持一致：网格边界用整数除法，避免分辨率除不尽时错位。
    ys = [h * r // rows for r in range(rows + 1)]
    xs = [w * c // cols for c in range(cols + 1)]
    for r in range(rows):
        y0, y1 = int(ys[r]), int(ys[r + 1])
        if y1 <= y0:
            continue
        for cidx in range(cols):
            x0, x1 = int(xs[cidx]), int(xs[cidx + 1])
            if x1 <= x0:
                continue
            cell = low[y0:y1, x0:x1]
            cm = valid[y0:y1, x0:x1]
            total = int(cell.size)
            n = int(np.count_nonzero(cm))
            cover[r, cidx] = n / float(total) if total else 0.0
            if n:
                vals = cell[cm].astype(np.float64)
                means[r, cidx] = float(vals.mean())
                stds[r, cidx] = float(vals.std())
    return means, stds, cover


def tone_grid(gray: np.ndarray, mask: np.ndarray,
              rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """每格低频均值；被 mask 完全遮住的格为 nan。"""
    g = lowpass(gray)
    out = np.full((max(1, int(rows)), max(1, int(cols))), np.nan, np.float64)
    if g.size == 0:
        return out
    h, w = g.shape[:2]
    if mask is None:
        valid = np.ones((h, w), bool)
    else:
        m = _norm_gray(mask)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        valid = m > 127
    means, _, _ = _grid_stats(g.astype(np.float32), valid, out.shape[0], out.shape[1])
    return means


# ---------------------------------------------------------------------------
# 相邻格合并（简单并查集）
# ---------------------------------------------------------------------------
def _merge_cell_groups(cells: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """把同 code 的上下左右相邻格合成一组。"""
    parent = {cell: cell for cell in cells}

    def find(x: Tuple[int, int]) -> Tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cellset = set(cells)
    for (r, c) in cellset:
        for dr, dc in ((1, 0), (0, 1)):
            nb = (r + dr, c + dc)
            if nb in cellset:
                union((r, c), nb)

    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for cell in cellset:
        groups.setdefault(find(cell), []).append(cell)
    return [sorted(g) for g in groups.values()]


def _group_label(cells: Sequence[Tuple[int, int]]) -> str:
    """'上左区' + '上中左区' -> '上左+上中左区'。"""
    labels = [cell_label(r, c) for (r, c) in cells]
    if len(labels) == 1:
        return labels[0]
    if len(labels) <= 6:
        return f"{labels[0][:-1]}+" + "+".join(labels[1:])
    return f"{labels[0][:-1]}+{labels[1][:-1]}+…（共{len(labels)}格）"


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------
def compare_value(ref_gray: np.ndarray, copy_gray: np.ndarray,
                  valid: Optional[np.ndarray] = None,
                  min_conf: float = 0.30) -> List[Finding]:
    """在低频图上逐格比较明暗，返回最多 4 条 Finding；空/全无效返回 []。"""
    rg = _norm_gray(ref_gray)
    cg = _norm_gray(copy_gray)
    if rg.size == 0 or cg.size == 0:
        return []
    if rg.shape != cg.shape:
        cg = cv2.resize(cg, (rg.shape[1], rg.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = rg.shape[:2]
    if h <= 0 or w <= 0:
        return []

    if valid is None:
        vmask = np.ones((h, w), bool)
    else:
        vm = _norm_gray(valid)
        if vm.shape[:2] != (h, w):
            vm = cv2.resize(vm, (w, h), interpolation=cv2.INTER_NEAREST)
        vmask = vm > 127
    if not vmask.any():
        return []

    rows, cols = GRID_ROWS, GRID_COLS
    rl = lowpass(rg).astype(np.float32)
    cl = lowpass(cg).astype(np.float32)
    rm, rs, rc = _grid_stats(rl, vmask, rows, cols)
    cm_, cs, cc = _grid_stats(cl, vmask, rows, cols)
    # 峰值：>0 表示临摹在该格更暗
    dmap = rl - cl
    pk_dark = _grid_peak(dmap, vmask, rows, cols)
    pk_bright = _grid_peak(-dmap, vmask, rows, cols)

    usable = ((rc >= MIN_VALID_COVERAGE) & (cc >= MIN_VALID_COVERAGE)
              & np.isfinite(rm) & np.isfinite(cm_)
              & np.isfinite(rs) & np.isfinite(cs))
    diff = cm_ - rm
    ratio_std = cs / np.maximum(rs, 1e-6)

    dark_cells = [(r, c) for r in range(rows) for c in range(cols)
                  if usable[r, c] and diff[r, c] < -DARK_DELTA]
    bright_cells = [(r, c) for r in range(rows) for c in range(cols)
                    if usable[r, c] and diff[r, c] > BRIGHT_DELTA]
    contrast_cells = [(r, c) for r in range(rows) for c in range(cols)
                      if usable[r, c] and rs[r, c] > CONTRAST_MIN_STD
                      and ratio_std[r, c] < CONTRAST_RATIO]

    candidates: List[Finding] = []
    for code, cells in (("VAL_DARK", dark_cells),
                        ("VAL_BRIGHT", bright_cells),
                        ("VAL_CONTRAST", contrast_cells)):
        pk = pk_dark if code == "VAL_DARK" else (
            pk_bright if code == "VAL_BRIGHT" else None)
        group_findings = [
            f for f in (_make_finding(code, g, diff, rm, cm_, rs, cs, rc, cc, pk)
                        for g in _merge_cell_groups(cells))
            if f is not None and f.confidence >= min_conf
        ]
        group_findings.sort(key=lambda f: f.severity, reverse=True)
        candidates.extend(group_findings[:MAX_PER_CODE])

    candidates.sort(key=lambda f: f.severity, reverse=True)
    return candidates[:MAX_FINDINGS]


def _grid_peak(arr: np.ndarray, vmask: np.ndarray,
               rows: int, cols: int) -> np.ndarray:
    """逐格取最大正值；没有有效像素的格为 0。

    格均值会把"只覆盖半格的一大块压暗"摊薄（压暗 30 级只读出 13 级），
    所以同时记录峰值，报告里两个数都写出来。
    """
    h, w = arr.shape[:2]
    peak = np.zeros((rows, cols), np.float64)
    ys = [h * r // rows for r in range(rows + 1)]
    xs = [w * c // cols for c in range(cols + 1)]
    for r in range(rows):
        for c in range(cols):
            blk = arr[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            msk = vmask[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            if blk.size and msk.any():
                v = blk[msk]
                if v.size:
                    peak[r, c] = float(np.max(v))
    return peak


def _make_finding(code: str, cells: List[Tuple[int, int]],
                  diff: np.ndarray, rm: np.ndarray, cm_: np.ndarray,
                  rs: np.ndarray, cs: np.ndarray,
                  rc: np.ndarray, cc: np.ndarray,
                  peak: Optional[np.ndarray] = None) -> Optional[Finding]:
    if not cells:
        return None

    # 取偏差最大的格做 region 锚点，位置文字仍列出整组。
    if code == "VAL_CONTRAST":
        idx = min(cells, key=lambda cell: cs[cell] / max(rs[cell], 1e-6))
    elif code == "VAL_DARK":
        idx = min(cells, key=lambda cell: diff[cell])
    else:
        idx = max(cells, key=lambda cell: diff[cell])
    r, c = idx
    cov = float(np.mean([rc[cell] for cell in cells] + [cc[cell] for cell in cells]))
    position = _group_label(cells)
    region = (r, c)
    cell_diff = float(diff[r, c])
    ref_mean = float(rm[r, c])
    copy_mean = float(cm_[r, c])
    ref_std = float(rs[r, c])
    copy_std = float(cs[r, c])
    ratio = copy_std / max(ref_std, 1e-6)
    pk = float(peak[r, c]) if peak is not None else None

    if code == "VAL_DARK":
        mag = abs(cell_diff)
        sev = 30.0 + 60.0 * min(1.0, max(0.0, mag - DARK_DELTA) / 40.0)
        conf = clamp01(0.35 + 0.45 * min(1.0, mag / 40.0)
                       + 0.20 * min(1.0, cov))
        if pk is not None and pk > mag + 3.0:
            qty = (f"该区低频均值 {copy_mean:.0f}，样张 {ref_mean:.0f}"
                   f"（该区平均暗 {mag:.0f} 级，最深 {pk:.0f} 级）")
        else:
            qty = (f"该区低频均值 {copy_mean:.0f}，样张 {ref_mean:.0f}"
                   f"（暗 {mag:.0f} 级）")
        return Finding(
            code=code, layer="value", severity=sev, position=position,
            phenomenon="该区整体比样张偏暗，暗部糊成一片。",
            quantity=qty,
            suggestion="该区整体减一层灰，保留最深处的暗，不要全压平",
            recheck="眯眼看该区，最暗处应仍比周围深一档",
            confidence=conf,
            metric={"cells": [[int(x), int(y)] for x, y in cells],
                    "mean_copy": round(copy_mean, 1),
                    "mean_ref": round(ref_mean, 1),
                    "delta": round(cell_diff, 1),
                    "peak": None if pk is None else round(pk, 1),
                    "coverage": round(cov, 3)},
            region=region,
        )

    if code == "VAL_BRIGHT":
        mag = abs(cell_diff)
        sev = 30.0 + 60.0 * min(1.0, max(0.0, mag - BRIGHT_DELTA) / 40.0)
        conf = clamp01(0.35 + 0.45 * min(1.0, mag / 40.0)
                       + 0.20 * min(1.0, cov))
        if pk is not None and pk > mag + 3.0:
            qty = (f"该区低频均值 {copy_mean:.0f}，样张 {ref_mean:.0f}"
                   f"（该区平均亮 {mag:.0f} 级，最亮处 {pk:.0f} 级）")
        else:
            qty = (f"该区低频均值 {copy_mean:.0f}，样张 {ref_mean:.0f}"
                   f"（亮 {mag:.0f} 级）")
        return Finding(
            code=code, layer="value", severity=sev, position=position,
            phenomenon="该区整体比样张偏亮，该压的明暗没压下去。",
            quantity=qty,
            suggestion="该区整体压暗一层，留住纸白只放在最亮处，中间调跟着降",
            recheck="眯眼看该区，最亮处不应和纸白一样亮",
            confidence=conf,
            metric={"cells": [[int(x), int(y)] for x, y in cells],
                    "mean_copy": round(copy_mean, 1),
                    "mean_ref": round(ref_mean, 1),
                    "delta": round(cell_diff, 1),
                    "peak": None if pk is None else round(pk, 1),
                    "coverage": round(cov, 3)},
            region=region,
        )

    # VAL_CONTRAST
    deficit = max(0.0, CONTRAST_RATIO - ratio)
    sev = 25.0 + 55.0 * min(1.0, deficit / 0.35)
    conf = clamp01(0.35 + 0.45 * min(1.0, deficit / 0.35)
                   + 0.20 * min(1.0, cov))
    return Finding(
        code=code, layer="value", severity=sev, position=position,
        phenomenon="该区发灰，明暗对比不足，层次并成一片。",
        quantity=(f"格内标准差 {copy_std:.1f}，样张 {ref_std:.1f}"
                  f"（低 {(1.0 - ratio) * 100:.0f}%）"),
        suggestion="先定最暗点再定最亮点，中间调只留两个层次",
        recheck="眯眼后该区应仍有清晰的黑白灰三段",
        confidence=conf,
        metric={"cells": [[int(x), int(y)] for x, y in cells],
                "std_copy": round(copy_std, 2),
                "std_ref": round(ref_std, 2),
                "std_ratio": round(ratio, 3),
                "coverage": round(cov, 3)},
        region=region,
    )


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H = W = 400
    xs = np.linspace(40, 220, W, dtype=np.float32)
    ys = np.linspace(0, 30, H, dtype=np.float32)
    ref = np.clip(xs[None, :] + ys[:, None], 0, 255).astype(np.uint8)
    copy = ref.copy()
    # 右下 2x2 格（像素 200:400, 200:400）整体压暗 30 级
    block = copy[200:400, 200:400].astype(np.int16) - 30
    copy[200:400, 200:400] = np.clip(block, 0, 255).astype(np.uint8)
    valid = np.full((H, W), 255, np.uint8)

    findings = compare_value(ref, copy, valid)
    codes = [f.code for f in findings]
    assert "VAL_DARK" in codes, f"未检出 VAL_DARK: {codes}"
    assert len(findings) <= 4, f"总条数超过 4: {codes}"
    dark = [f for f in findings if f.code == "VAL_DARK"]
    assert len(dark) == 1, f"相邻暗格应合并为 1 条，实际 {len(dark)} 条"
    assert dark[0].region[0] >= 2 and dark[0].region[1] >= 2, \
        f"region 应落在右下，实际 {dark[0].region}"
    # 未改动区不得误报
    for f in findings:
        assert f.region[0] >= 2 and f.region[1] >= 2, \
            f"{f.code} 误报在 region={f.region} position={f.position}"
    # 每类最多 2 条、总共最多 4 条
    for code in ("VAL_DARK", "VAL_BRIGHT", "VAL_CONTRAST"):
        assert sum(1 for f in findings if f.code == code) <= 2

    # tone_grid：右下 2x2 平均应比同位置 ref 低约 30
    tg_ref = tone_grid(ref, valid)
    tg_copy = tone_grid(copy, valid)
    assert tg_ref.shape == (4, 4)
    assert abs((tg_ref[2, 2] - tg_copy[2, 2]) - 30.0) < 3.0, \
        (tg_ref[2, 2], tg_copy[2, 2])
    # 无效格为 nan
    tg_masked = tone_grid(ref, np.zeros((H, W), np.uint8))
    assert np.isnan(tg_masked).all()

    # 空 / 纯色 / 全无效：不崩，返回空
    flat = np.full((120, 120), 128, np.uint8)
    assert compare_value(flat, flat, None) == []
    assert compare_value(np.zeros((0, 0), np.uint8),
                         np.zeros((0, 0), np.uint8), None) == []
    assert compare_value(flat, flat, np.zeros((120, 120), np.uint8)) == []
    assert compare_value(flat, np.full((120, 120), 160, np.uint8), None)  # 均匀偏亮可报
    assert abs(ssim(flat, flat) - 1.0) < 1e-6
    assert 0.0 < ssim(flat, np.full((120, 120), 88, np.uint8)) < 1.0
    assert lowpass(flat).shape == flat.shape
    assert lowpass(np.zeros((0, 0), np.uint8)).size == 0

    print("[value] PASS")
    print("  codes:", codes)
    for f in findings:
        print(f"  {f.code}: {f.quantity} | position={f.position} region={f.region}")
