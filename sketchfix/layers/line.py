"""线稿/轮廓层：Zhang-Suen 细化 + 双向 chamfer 距离比对。

模块职责
--------
1. ``thin``            —— 纯 numpy 向量化的 Zhang-Suen 细化（无 cv2.ximgproc）。
2. ``skeletonize``     —— 灰度图 -> 墨迹 mask -> 3x3 close -> 1px 骨架。
3. ``chamfer_stats``   —— 双向 chamfer 距离统计（缺线 / 多线的原始量化）。
4. ``compare_lines``   —— 4x4 网格内判定 LINE_MISSING / LINE_EXTRA /
                          LINE_SHIFT，并在全局补一条 LINE_DENSITY。

约定
----
* ``ref_gray`` / ``copy_gray``：uint8 灰度，已对齐（copy 已 warp 到 ref 画布）。
* 所有 mask：uint8，0 或 255，255 = 前景/墨迹。
* 距离单位：像素（``cv2.DIST_L2``，5x5 掩膜）。

阈值依据
--------
网格边长为图宽的 1/4、图高的 1/4；「看得见的偏差」在整幅图上约为对角线的
1.2%（1400px 长边的图约 17px，800x600 的图约 12px）。小图上下限取 6px，
避免 6px 以下的抖动被判成错误。判定偏移时用 0.010*diag 作门槛（略低于
缺/多线门槛），因为偏移是双向同时变大的整体现象，单侧误差会互相抵消。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..types import GRID_COLS, GRID_ROWS, Finding, cell_label, clamp01, odd_kernel, to_gray

# 距离统计用到的固定门槛（像素）：
#   miss_mean / extra_mean < 6 才允许判「缺线 / 多线」——若双向都大，那是整体偏移
#   而不是某一侧的问题（见 LINE_SHIFT）。
BIDIR_GUARD = 6.0
# 反向距离不到正向的 35% 时，即使超过 6px 也算"反向干净"：缺一条腿时，
# 反向均值会被附近的别的线抬到 6~10px，按绝对值卡会漏检。
BIDIR_RATIO = 0.35


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _as_mask(m: Any, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """任意输入 -> uint8 0/255 单通道 mask；形状不符时用最近邻缩放。"""
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = to_gray(arr)
    if arr.ndim != 2:
        raise ValueError(f"mask 必须是 2D，得到 shape={arr.shape}")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        f = arr.astype(np.float32)
        if f.size and float(f.max()) <= 1.0:  # 0/1 的浮点 mask 先拉满
            f *= 255.0
        arr = np.clip(f, 0, 255).astype(np.uint8)
    out = np.where(arr > 127, 255, 0).astype(np.uint8)
    if shape is not None and out.shape != shape:
        out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.where(out > 127, 255, 0).astype(np.uint8)
    return out


def _dt(skel: np.ndarray) -> np.ndarray:
    """到最近骨架点的欧氏距离场。

    ``cv2.distanceTransform`` 求的是「到最近 0 像素」的距离，所以喂 ``255 - skel``。
    空骨架时 distanceTransform 没有 0 像素可用，这里补一个以对角线为上限的
    常数场，保证「整块没画」也能被判成缺线（而不是得到全 0）。
    """
    s = _as_mask(skel)
    h, w = s.shape
    if not s.any():
        return np.full((h, w), float(np.hypot(h, w)), dtype=np.float32)
    return cv2.distanceTransform(255 - s, cv2.DIST_L2, 5)


def _stats_from_values(vals: np.ndarray) -> Dict[str, float]:
    """一组距离值 -> mean/p95/max/n（空集全 0，n=0）。"""
    v = np.asarray(vals, dtype=np.float32).ravel()
    if v.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(v.mean()),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
        "n": int(v.size),
    }


def _severity(v: float, thr: float) -> float:
    """超阈倍数 -> 0-100 严重度：刚好到阈 40 分，2 倍阈 100 分。"""
    if thr <= 1e-6:
        return 100.0
    return float(min(100.0, 40.0 + 60.0 * (v / thr - 1.0)))


def _confidence(v: float, thr: float, n: float) -> float:
    """置信度看"证据有多硬"而不是"这格有多少墨"。

    早期用 ink_ratio 推置信度，结果画得细的格子（墨迹占比 2%）即使偏离
    200px 也只有 0.37 分（显示成"低，建议人工复核"），而判据本身很硬。
    改为：偏离超过阈值越多、采样点越多 -> 越确定。
    """
    if thr <= 1e-6:
        return 0.9
    over = max(0.0, min(1.0, (v / thr - 1.0) / 3.0))
    pts = max(0.0, min(1.0, (n - 3.0) / 40.0))
    return float(min(0.92, clamp01(0.40 + 0.35 * over + 0.15 * pts)))


def _ink_mask(gray: np.ndarray, fg: Optional[np.ndarray] = None) -> np.ndarray:
    """灰度图 -> 墨迹 mask。

    * Otsu 全局阈值取暗部（THRESH_BINARY_INV）；给了 fg 就只保留 fg 内的暗部。
    * 自适应阈值（高斯，31x31，C=10）补回浅色/局部对比低的笔画。
    * 两者取并集后 3x3 close，把断口和抗锯齿毛刺接起来。
    """
    g = to_gray(gray)
    h, w = g.shape
    shape = (h, w)
    _t, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    otsu = np.where(otsu > 0, 255, 0).astype(np.uint8)

    bs = odd_kernel(31)
    if min(h, w) < bs + 2:  # 太小的图收缩 blockSize，cv2 要求 < 图像尺寸
        bs = odd_kernel(max(3, min(h, w) - 1))
    adaptive = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 10
    )
    mask = cv2.bitwise_or(otsu, adaptive)

    if fg is not None:
        fgm = _as_mask(fg, shape)
        mask = cv2.bitwise_and(mask, fgm)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return np.where(mask > 0, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 1. 细化
# --------------------------------------------------------------------------- #
def thin(binary: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Zhang-Suen 细化，输入/输出 uint8 0/255。

    全部用切片向量化（每个子步 8 个邻域数组），不做 python 逐像素循环；
    1400px 级别的图也在 1s 内收敛。两遍扫描（step 0 / step 1）交替进行，
    直到某一轮没有任何像素被删或达到 ``max_iter``（默认 60）。
    """
    img = np.asarray(binary)
    if img.ndim != 2:
        img = to_gray(img)
    src = np.where(img > 0, 255, 0).astype(np.uint8)
    if not src.any():
        return np.zeros_like(src)

    work = (src > 0).astype(np.uint8)

    for _ in range(max(1, int(max_iter))):
        removed = False
        for step in (0, 1):
            # 1px 补边后切片取 8 邻域：p2=上, p3=右上, p4=右, p5=右下,
            # p6=下, p7=左下, p8=左, p9=左上（顺时针，与经典论文一致）
            p = np.pad(work, 1, mode="constant")
            p1 = p[1:-1, 1:-1]
            p2 = p[:-2, 1:-1]
            p3 = p[:-2, 2:]
            p4 = p[1:-1, 2:]
            p5 = p[2:, 2:]
            p6 = p[2:, 1:-1]
            p7 = p[2:, :-2]
            p8 = p[1:-1, :-2]
            p9 = p[:-2, :-2]
            nb = (p2, p3, p4, p5, p6, p7, p8, p9)

            b = np.zeros(p1.shape, np.uint8)
            for n in nb:
                b += n
            # A: 8 邻域绕一圈的 0->1 跳变次数；骨架端点的充要条件是 A == 1
            a = np.zeros(p1.shape, np.uint8)
            for k in range(8):
                a += ((nb[k] == 0) & (nb[(k + 1) % 8] == 1)).astype(np.uint8)

            if step == 0:
                c1 = (p2 * p4 * p6) == 0
                c2 = (p4 * p6 * p8) == 0
            else:
                c1 = (p2 * p4 * p8) == 0
                c2 = (p2 * p6 * p8) == 0

            cond = (p1 == 1) & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
            if cond.any():
                work[cond] = 0
                removed = True

        if not removed:
            break

    out = (work * 255).astype(np.uint8)
    # 结果必须是原图墨迹的子集（防补边/边界带来的意外像素）
    return cv2.bitwise_and(out, src)


# --------------------------------------------------------------------------- #
# 2. 骨架化
# --------------------------------------------------------------------------- #
def skeletonize(gray: np.ndarray, fg: Optional[np.ndarray] = None) -> np.ndarray:
    """灰度图 -> 1px 骨架（uint8 0/255）。

    墨迹 = Otsu 暗部（fg 内）∪ 自适应阈值，3x3 close 后交给 ``thin``。
    """
    return thin(_ink_mask(gray, fg))


# --------------------------------------------------------------------------- #
# 3. chamfer 统计
# --------------------------------------------------------------------------- #
def chamfer_stats(
    a_skel: np.ndarray,
    b_skel: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """双向 chamfer 距离统计。

    参数
    ----
    a_skel, b_skel : 两幅骨架（uint8 0/255）
    valid          : 有效区 mask，None 视为全图有效

    返回
    ----
    ``miss_*``  : 以 a 为参考采样到 b 的距离（a 有线、b 没有 = 缺线）
    ``extra_*`` : 以 b 为采样点到 a 的距离（b 多出来的线）
    每项含 mean / p95 / max / n（采样点数），另含 ``diag`` 与 ``valid_ratio``。
    """
    a = _as_mask(a_skel)
    b = _as_mask(b_skel, a.shape)
    if valid is None:
        v = np.full(a.shape, 255, np.uint8)
    else:
        v = _as_mask(valid, a.shape)

    d_to_b = _dt(b)   # 采样在 a 上 -> a 有、b 缺
    d_to_a = _dt(a)   # 采样在 b 上 -> b 多出
    sa = (a > 0) & (v > 0)
    sb = (b > 0) & (v > 0)

    miss = _stats_from_values(d_to_b[sa])
    extra = _stats_from_values(d_to_a[sb])
    h, w = a.shape
    out: Dict[str, Any] = {
        "miss_mean": miss["mean"], "miss_p95": miss["p95"],
        "miss_max": miss["max"], "miss_n": miss["n"],
        "extra_mean": extra["mean"], "extra_p95": extra["p95"],
        "extra_max": extra["max"], "extra_n": extra["n"],
        "diag": float(np.hypot(h, w)),
        "valid_ratio": float((v > 0).mean()),
    }
    return out


# --------------------------------------------------------------------------- #
# 4. Finding 组装
# --------------------------------------------------------------------------- #
def _mk_missing(r: int, c: int, ms: Dict[str, float], thr: float,
                ink_ratio: float) -> Finding:
    return Finding(
        code="LINE_MISSING",
        layer="line",
        severity=_severity(ms["p95"], thr),
        position=cell_label(r, c),
        region=(r, c),
        phenomenon="该区样张有线的位置临摹没有跟上，轮廓断开或整段缺失。",
        quantity=(
            f"该区缺线处平均偏离 {ms['mean']:.1f}px、95% 分位 {ms['p95']:.1f}px"
            f"（阈值 {thr:.0f}px，参考骨架点 {ms['n']} 个）"
        ),
        suggestion=(
            "先补该区最长的一条外轮廓，再补齐内部分界；从样张上该区线条的一端起笔，"
            "顺着一笔画到另一端，中途不要抬笔，避免又断成两段。"
        ),
        recheck=(
            "遮住样张只看临摹，该区轮廓是否断成两段；再对着样张比对该区线条的起止点，"
            "端点应重合、线宽应一致。"
        ),
        confidence=_confidence(ms["p95"], thr, ms["n"]),
        metric={
            "miss_mean": round(ms["mean"], 2),
            "miss_p95": round(ms["p95"], 2),
            "miss_n": ms["n"],
            "miss_thr": round(thr, 2),
            "ink_ratio": round(ink_ratio, 4),
        },
    )


def _mk_extra(r: int, c: int, es: Dict[str, float], thr: float,
              ink_ratio: float) -> Finding:
    return Finding(
        code="LINE_EXTRA",
        layer="line",
        severity=_severity(es["p95"], thr),
        position=cell_label(r, c),
        region=(r, c),
        phenomenon="该区多出样张上没有的线条，空白处被多余的线穿过。",
        quantity=(
            f"该区多余线条平均偏离 {es['mean']:.1f}px、95% 分位 {es['p95']:.1f}px"
            f"（阈值 {thr:.0f}px，多余骨架点 {es['n']} 个）"
        ),
        suggestion=(
            "擦掉该区多出的线条（偏离样张轮廓的那一段），只保留样张上也有的轮廓；"
            "若那是辅助线/草稿线，先擦净再补主线，别用深线覆盖。"
        ),
        recheck=(
            "把样张叠在临摹上对着光看，该区空白处是否有多余的线穿过；"
            "再单看临摹，该区线条数量是否与样张一致。"
        ),
        confidence=_confidence(es["p95"], thr, es["n"]),
        metric={
            "extra_mean": round(es["mean"], 2),
            "extra_p95": round(es["p95"], 2),
            "extra_n": es["n"],
            "extra_thr": round(thr, 2),
            "ink_ratio": round(ink_ratio, 4),
        },
    )


def _mk_shift(r: int, c: int, ms: Dict[str, float], es: Dict[str, float],
              thr: float, ink_ratio: float) -> Finding:
    return Finding(
        code="LINE_SHIFT",
        layer="line",
        severity=_severity(ms["mean"], thr),
        position=cell_label(r, c),
        region=(r, c),
        phenomenon="轮廓整体偏移：样张与临摹的线平行错开，形状没大问题但位置不对。",
        quantity=(
            f"该区轮廓双向平均偏离 样张->临摹 {ms['mean']:.1f}px、"
            f"临摹->样张 {es['mean']:.1f}px（阈值 {thr:.0f}px），偏移量约 {ms['mean']:.1f}px"
        ),
        suggestion=(
            f"不要重画线条形状，把该区整块轮廓朝样张方向平移约 {ms['mean']:.1f}px："
            "先选该区最长的一条线对齐，其余线条跟着整体平移，保持线间距不变。"
        ),
        recheck=(
            "遮住样张只看临摹，该区轮廓走向是否与样张平行；再用直尺量该区两条对应长线的"
            "间距，上下、左右各量一次，间距应处处相等。"
        ),
        confidence=_confidence(ms["mean"], thr, min(ms["n"], es["n"])),
        metric={
            "miss_mean": round(ms["mean"], 2),
            "extra_mean": round(es["mean"], 2),
            "shift_thr": round(thr, 2),
            "ratio": round(ms["mean"] / max(es["mean"], 1e-6), 3),
            "ink_ratio": round(ink_ratio, 4),
        },
    )


# --------------------------------------------------------------------------- #
# 5. 主比对
# --------------------------------------------------------------------------- #
def compare_lines(
    ref_gray: np.ndarray,
    copy_gray: np.ndarray,
    valid: np.ndarray,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
    min_conf: float = 0.30,
) -> List[Finding]:
    """按网格比对线稿：缺线 / 多线 / 偏移 / 全局密度。

    参数
    ----
    ref_gray, copy_gray : 已对齐的 uint8 灰度（copy 已 warp 到 ref 画布）
    valid               : 有效区 mask（0/255）；临摹纸外的区域不参与统计
    rows, cols          : 网格行列（默认 4x4）
    min_conf            : 置信度下限，低于此值的 finding 丢弃

    统计口径
    --------
    只在 ``valid`` 内、且该格样张有墨（``ink_ratio >= 0.005``）时判定 —— 空白
    格不判，避免「样张本来没画」被误报成缺线或多线。距离场在整幅图上算一次，
    再按格采样，避免格边界处的距离被截断。
    """
    ref = to_gray(ref_gray)
    copy = to_gray(copy_gray)
    if copy.shape != ref.shape:
        copy = cv2.resize(copy, (ref.shape[1], ref.shape[0]),
                          interpolation=cv2.INTER_AREA)
    h, w = ref.shape
    shape = (h, w)

    v = _as_mask(valid, shape) if valid is not None else np.full(shape, 255, np.uint8)
    valid_bool = v > 0

    ref_ink = _ink_mask(ref)
    copy_ink = _ink_mask(copy)
    ref_skel = thin(ref_ink)
    copy_skel = thin(copy_ink)

    # 整幅距离场：d_ref 采样在 ref 骨架上 = 缺线；d_copy 采样在 copy 骨架上 = 多线
    d_ref = _dt(copy_skel)
    d_copy = _dt(ref_skel)

    diag = float(np.hypot(h, w))
    miss_thr = max(6.0, 0.012 * diag)
    extra_thr = max(6.0, 0.012 * diag)
    shift_thr = max(4.0, 0.010 * diag)

    findings: List[Finding] = []
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    for r in range(rows):
        y0, y1 = h * r // rows, h * (r + 1) // rows
        for c in range(cols):
            x0, x1 = w * c // cols, w * (c + 1) // cols
            sl = (slice(y0, y1), slice(x0, x1))
            cell_v = valid_bool[sl]
            n_valid = int(cell_v.sum())
            if n_valid == 0:
                continue
            ink = int((ref_ink[sl] > 0)[cell_v].sum())
            ink_ratio = ink / float(n_valid)
            if ink_ratio < 0.005:
                # 样张空白格：默认不判（避免纸张纹理误报）。但临摹在这里画出了
                # 明显的线，属于"无中生有"，必须报——否则删掉/多画的线会漏检。
                copy_ratio = float((copy_ink[sl] > 0)[cell_v].sum()) / float(n_valid)
                e_mask0 = (copy_skel[sl] > 0) & cell_v
                es0 = _stats_from_values(d_copy[sl][e_mask0])
                if copy_ratio > 0.008 and es0["n"] >= 8 and es0["p95"] > extra_thr:
                    f0 = _mk_extra(r, c, es0, extra_thr, copy_ratio)
                    f0.phenomenon = "样张该区基本空白，临摹却多画了线条。"
                    f0.suggestion = (
                        "先确认这条线是不是多余的：如果是辅助线/投影/衣皱的臆造，擦掉；"
                        "如果是刻意加的，就保留但需要在别处交代它的由来。"
                    )
                    f0.metric["ref_blank"] = True
                    findings.append(f0)
                continue

            m_mask = (ref_skel[sl] > 0) & cell_v
            e_mask = (copy_skel[sl] > 0) & cell_v
            ms = _stats_from_values(d_ref[sl][m_mask])
            es = _stats_from_values(d_copy[sl][e_mask])

            cands: List[Finding] = []
            # 「整体偏移」的判别不能只看绝对值：真缺一条腿时反向均值也可能到
            # 6px（附近有别的线），所以再加一条相对判据——反向远小于正向才是
            # 单向缺失。
            # 缺线：单向（ref->copy）p95 超阈，且反向干净 -> 不是整体偏移
            if (ms["n"] >= 3 and ms["p95"] > miss_thr
                    and es["mean"] < max(BIDIR_GUARD, BIDIR_RATIO * ms["mean"])):
                cands.append(_mk_missing(r, c, ms, miss_thr, ink_ratio))
            # 多线：单向（copy->ref）p95 超阈，且正向干净
            if (es["n"] >= 3 and es["p95"] > extra_thr
                    and ms["mean"] < max(BIDIR_GUARD, BIDIR_RATIO * es["mean"])):
                cands.append(_mk_extra(r, c, es, extra_thr, ink_ratio))
            # 偏移：双向均值都大且量级接近（0.6~1.6），形状对但整体错位
            if (ms["n"] >= 3 and es["n"] >= 3
                    and ms["mean"] > shift_thr and es["mean"] > shift_thr):
                ratio = ms["mean"] / max(es["mean"], 1e-6)
                if 0.6 <= ratio <= 1.6:
                    cands.append(_mk_shift(r, c, ms, es, shift_thr, ink_ratio))
            if cands:                  # 每格最多 1 条，取最严重者
                findings.append(max(cands, key=lambda f: f.severity))

    # ---- 全局墨量密度：线条过密/过疏，往往是一类系统性错误 ----
    ref_ink_n = int((ref_ink > 0)[valid_bool].sum())
    copy_ink_n = int((copy_ink > 0)[valid_bool].sum())
    if ref_ink_n >= 50:
        density = copy_ink_n / float(ref_ink_n)
        dev = abs(density - 1.0)
        if dev > 0.25:
            direction = "偏多" if density > 1.0 else "偏少"
            global_ratio = ref_ink_n / float(max(1, int(valid_bool.sum())))
            findings.append(Finding(
                code="LINE_DENSITY",
                layer="line",
                severity=float(min(100.0, 40.0 + 60.0 * (dev / 0.25 - 1.0))),
                position="整幅",
                region=(-1, -1),
                phenomenon=f"整体线条{direction}：临摹的总墨量与样张明显不符。",
                quantity=(
                    f"临摹墨量 {copy_ink_n}px、样张墨量 {ref_ink_n}px，"
                    f"偏差 {(density - 1.0) * 100:+.1f}%（阈值 ±25%）"
                ),
                suggestion=(
                    f"线条{direction}是整幅性问题，先别看局部：临摹是否把辅助线/排线也留着了、"
                    f"或线画得比样张粗。先统一线宽，再擦掉样张上没有的线"
                    f"（或补上漏掉的整段轮廓），不要靠加深颜色掩盖。"
                ) if density > 1.0 else (
                    "线条偏少是整幅性问题：先检查是否用了过细的笔或过浅的力度，"
                    "再把样张上每条主轮廓都描一遍，优先补外轮廓。"
                ),
                recheck=(
                    "把样张和临摹并排放在同一距离看，两幅的「黑度」是否接近；"
                    "再抽查 2-3 个区域，数一数线条条数是否一致。"
                ),
                confidence=min(0.9, clamp01(0.35 + global_ratio * 0.8)),
                metric={
                    "ref_ink_px": ref_ink_n,
                    "copy_ink_px": copy_ink_n,
                    "density_ratio": round(density, 3),
                    "density_dev": round(dev, 3),
                },
            ))

    out = [f for f in findings if f.confidence >= float(min_conf)]
    out.sort(key=lambda f: f.severity, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _demo_images():
    """800x600 合成图：圆头 + 梯形身体 + 四肢；删右腿、多一条横线。"""
    h, w = 600, 800
    ref = np.full((h, w), 255, np.uint8)
    ink = 0
    cv2.circle(ref, (400, 100), 55, ink, 3)                    # 圆头
    cv2.line(ref, (400, 155), (400, 172), ink, 3)              # 脖子
    body = np.array([[355, 170], [445, 170], [490, 360], [310, 360]], np.int32)
    cv2.polylines(ref, [body], True, ink, 3)                   # 梯形身体
    cv2.line(ref, (355, 182), (258, 300), ink, 3)              # 左臂
    cv2.line(ref, (445, 182), (542, 300), ink, 3)              # 右臂
    cv2.line(ref, (350, 360), (298, 540), ink, 3)              # 左腿
    cv2.line(ref, (450, 360), (520, 540), ink, 3)              # 右腿

    copy = ref.copy()
    cv2.line(copy, (450, 360), (520, 540), 255, 11)            # 删掉右腿
    cv2.line(copy, (330, 250), (470, 250), ink, 3)             # 身体里多一条横线

    valid = np.full((h, w), 255, np.uint8)
    return ref, copy, valid


if __name__ == "__main__":
    import time

    print("== line.py 自检 ==")
    ref, copy, valid = _demo_images()

    # thin 性能：1000x750 墨迹掩膜（按规范 < 2s）
    big = cv2.resize(ref, (1000, 750), interpolation=cv2.INTER_NEAREST)
    big_ink = _ink_mask(big)
    t0 = time.perf_counter()
    big_skel = thin(big_ink)
    thin_dt = time.perf_counter() - t0
    print(f"thin(1000x750) 耗时 {thin_dt * 1000:.1f} ms，墨迹 {int((big_ink > 0).sum())}px、"
          f"骨架 {int((big_skel > 0).sum())}px")
    assert thin_dt < 2.0, f"thin 太慢: {thin_dt:.2f}s"
    assert 0 < int((big_skel > 0).sum()) < int((big_ink > 0).sum())

    # 细化基本性质：骨架是原墨迹（3x3 close 后）的子集
    ref_ink = _ink_mask(ref)
    skel = skeletonize(ref)
    assert int((skel > 0).sum()) > 0, "骨架为空"
    assert not ((skel > 0) & (ref_ink == 0)).any(), "骨架越出了墨迹范围"

    t1 = time.perf_counter()
    findings = compare_lines(ref, copy, valid)
    cmp_dt = time.perf_counter() - t1

    print(f"compare_lines 耗时 {cmp_dt * 1000:.1f} ms，检出 {len(findings)} 条")
    for f in findings:
        print(f"  {f.code:<12} region={f.region} pos={f.position:<6} "
              f"sev={f.severity:5.1f} conf={f.confidence:.2f} | {f.quantity}")

    missing = [f for f in findings if f.code == "LINE_MISSING"]
    extra = [f for f in findings if f.code == "LINE_EXTRA"]

    assert missing, "未检出 LINE_MISSING（右腿被删）"
    assert any(f.region[0] >= 2 and f.region[1] >= 2 for f in missing), \
        f"LINE_MISSING 位置不在右下方: {[f.region for f in missing]}"
    assert extra, "未检出 LINE_EXTRA（多画的横线）"

    # Finding 六项齐全
    for f in findings:
        for k in ("code", "layer", "severity", "position", "phenomenon",
                  "quantity", "suggestion", "recheck", "confidence"):
            val = getattr(f, k)
            assert val not in (None, ""), f"字段 {k} 为空: {f.code}"
        assert any(ch.isdigit() for ch in f.quantity), f"quantity 无数字: {f.quantity}"

    print(f"缺线 {len(missing)} 条 -> {[f.region for f in missing]}")
    print(f"多线 {len(extra)} 条 -> {[f.region for f in extra]}")
    print("PASS")
