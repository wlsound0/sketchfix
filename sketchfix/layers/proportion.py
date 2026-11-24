"""构图/比例层：从线稿 mask 提取实体剪影，测量比例指标，并与样张对比。

只依赖 ``sketchfix.types``。所有偏差阈值集中在本文件顶部，方便按画风调整。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..types import Finding, cell_label, clamp01, grid_cell, odd_kernel

# ---- 判定阈值 ----------------------------------------------------------
ASPECT_TOL = 0.08   # 长宽比相对样张偏差 > 8%
BBOX_TOL = 0.04     # 外接框单边偏差 > 画布宽/高的 4%
SHIFT_TOL = 0.03    # 质心位移 > 画布对角线的 3%
ANGLE_TOL = 3.0     # 主轴倾角差 > 3°
MAX_FINDINGS = 3    # 每类最多 1 条，总共最多 3 条，避免刷屏


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _as_mask(img: Any) -> Optional[np.ndarray]:
    """把各种输入的 mask 统一成 uint8 0/255；空输入返回 None。"""
    if img is None:
        return None
    a = np.asarray(img)
    if a.size == 0:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)
    if np.issubdtype(a.dtype, np.floating):
        a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0)
        if float(a.max()) <= 1.0:
            a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return np.where(a > 127, 255, 0).astype(np.uint8)


def _major_angle(contour: np.ndarray) -> float:
    """用图像矩二阶矩求主轴方向（度）。正角=屏幕坐标系顺时针。"""
    m = cv2.moments(contour)
    if m["m00"] <= 0:
        return 0.0
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    theta = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))
    # 归一到 (-90, 90]
    return float((theta + 90.0) % 180.0 - 90.0)


def _confidence(strength: float, area_ratio: float) -> float:
    """strength = 偏差/阈值；剪影占比越大越可信。"""
    s = min(1.5, max(0.0, float(strength)))
    return clamp01(0.45 + 0.35 * (s / 1.5) + 0.20 * min(1.0, area_ratio / 0.02))


# ---------------------------------------------------------------------------
# 剪影
# ---------------------------------------------------------------------------
def silhouette(fg: np.ndarray) -> np.ndarray:
    """由线稿 mask 得到实体剪影：闭运算连断线 + 填洞 + 只留最大连通域。"""
    mask = _as_mask(fg)
    if mask is None:
        return np.zeros((0, 0), np.uint8)
    if not mask.any():
        return np.zeros_like(mask)

    h, w = mask.shape[:2]
    # 长边/60 的椭圆核：1400px 图约 23px，足以跨过常见断线。
    k = odd_kernel(int(round(max(h, w) / 60.0)), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    # 跳过背景标签 0，取面积最大的实体，避免纸张碎边干扰。
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------
def measure(fg: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """测剪影的 bbox / 面积比 / 长宽比 / 质心 / 主轴角 / 填充率。

    空 mask（或有效区为零）返回 ``{"empty": True}``，调用方必须处理。
    """
    mask = _as_mask(fg)
    if mask is None or not mask.any():
        return {"empty": True}

    h, w = mask.shape[:2]
    vmask = _as_mask(valid)
    if vmask is not None:
        if vmask.shape[:2] != mask.shape[:2]:
            vmask = cv2.resize(vmask, (w, h), interpolation=cv2.INTER_NEAREST)
        valid_area = int(np.count_nonzero(vmask))
        if valid_area == 0:
            return {"empty": True}
        mask = cv2.bitwise_and(mask, vmask)
        if not mask.any():
            return {"empty": True}
    else:
        valid_area = int(w * h)
        if valid_area == 0:
            return {"empty": True}

    sil = silhouette(mask)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"empty": True}
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return {"empty": True}

    x, y, bw, bh = cv2.boundingRect(contour)
    area = float(np.count_nonzero(sil))
    m = cv2.moments(contour)
    if m["m00"] > 0:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
    else:
        cx, cy = x + bw / 2.0, y + bh / 2.0

    bbox_area = float(max(1, bw * bh))
    return {
        "empty": False,
        "bbox": (int(x), int(y), int(bw), int(bh)),
        "area_ratio": float(area / max(1, valid_area)),
        "aspect": float(bw / bh) if bh > 0 else 0.0,
        "centroid": (cx, cy),
        "angle_deg": _major_angle(contour),
        "fill": float(area / bbox_area),
        "area": area,
        "valid_area": int(valid_area),
        "canvas": (int(w), int(h)),
    }


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------
def compare_proportion(ref_fg: np.ndarray, copy_fg: np.ndarray,
                       valid: Optional[np.ndarray] = None,
                       min_conf: float = 0.35) -> List[Finding]:
    """比较样张与临摹的剪影比例，产出最多 3 条 Finding。"""
    ref_mask = _as_mask(ref_fg)
    if ref_mask is None or not ref_mask.any():
        return []
    H, W = ref_mask.shape[:2]
    if W <= 0 or H <= 0:
        return []

    # 保证 copy / valid 与 ref 同画布，坐标才可比。
    copy_mask = _as_mask(copy_fg)
    if copy_mask is None:
        return []
    if copy_mask.shape[:2] != (H, W):
        copy_mask = cv2.resize(copy_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    vmask = _as_mask(valid)
    if vmask is not None and vmask.shape[:2] != (H, W):
        vmask = cv2.resize(vmask, (W, H), interpolation=cv2.INTER_NEAREST)

    r = measure(ref_mask, vmask)
    c = measure(copy_mask, vmask)
    if r.get("empty") or c.get("empty"):
        return []

    area_ratio = min(float(r["area_ratio"]), float(c["area_ratio"]))
    diag = float(np.hypot(W, H))
    findings: List[Finding] = []

    # 1) 长宽比 ---------------------------------------------------------
    ar = float(r["aspect"])
    ac = float(c["aspect"])
    if ar > 1e-6 and ac > 1e-6:
        ratio = ac / ar
        rel = abs(ratio - 1.0)
        if rel > ASPECT_TOL:
            pct = rel * 100.0
            narrower = ratio < 1.0
            sev = 40.0 + 50.0 * min(1.0, (rel - ASPECT_TOL) / 0.20)
            findings.append(Finding(
                code="PROP_ASPECT",
                layer="proportion",
                severity=sev,
                position="整体构图",
                phenomenon=("整体轮廓比样张偏瘦长" if narrower
                            else "整体轮廓比样张偏宽扁"),
                quantity=(f"长宽比 {ac:.2f}，样张 {ar:.2f}"
                          f"（{ratio:.2f}×，{'窄' if narrower else '宽'} {pct:.0f}%）"),
                suggestion=(f"以最宽处为基准把轮廓{'放宽' if narrower else '收窄'}"
                            f"约 {pct:.0f}%，优先保住头身比与肩宽，再调四边留白"),
                recheck="叠放样张，比较最宽处与总高的比值是否一致",
                confidence=_confidence(rel / ASPECT_TOL, area_ratio),
                metric={"aspect_copy": round(ac, 4), "aspect_ref": round(ar, 4),
                        "aspect_ratio": round(ratio, 4)},
                region=grid_cell(c["centroid"][0], c["centroid"][1], W, H),
            ))

    # 2) 外接框单边 -----------------------------------------------------
    rx, ry, rw, rh = r["bbox"]
    cx0, cy0, cw, ch = c["bbox"]
    # (名称, copy 边坐标, ref 边坐标, 对应画布边长)
    edges = [
        ("上", cy0, ry, H),
        ("下", cy0 + ch, ry + rh, H),
        ("左", cx0, rx, W),
        ("右", cx0 + cw, rx + rw, W),
    ]
    edge_name, edge_copy, edge_ref, edge_denom = max(
        edges, key=lambda t: abs(t[1] - t[2]) / max(1.0, float(t[3])))
    dev = float(edge_copy - edge_ref)
    rel = abs(dev) / max(1.0, float(edge_denom))
    if rel > BBOX_TOL:
        if edge_name in ("上", "下"):
            word = "低" if dev > 0 else "高"
            move = "上" if dev > 0 else "下"
            axis = "竖直"
        else:
            word = "偏右" if dev > 0 else "偏左"
            move = "左" if dev > 0 else "右"
            axis = "水平"
        # region 统一取 copy 外接框中心所在格（spec 约定）；但外接框属于整体
        # 构图，文字位置不写某格名，免得读者以为只是角落那块的问题。
        bbox_region = grid_cell(cx0 + cw / 2.0, cy0 + ch / 2.0, W, H)
        sev = 30.0 + 55.0 * min(1.0, (rel - BBOX_TOL) / 0.16)
        findings.append(Finding(
            code="PROP_BBOX",
            layer="proportion",
            severity=sev,
            position=f"外框{edge_name}边",
            phenomenon=f"轮廓外接框的{edge_name}边界与样张不一致",
            quantity=f"{edge_name}边界比样张{word} {abs(dev):.0f}px（{rel * 100:.1f}%）",
            suggestion=(f"沿{axis}方向把{edge_name}边界向{move}移约 "
                        f"{abs(dev):.0f}px，其余三边先保持不动"),
            recheck="叠放样张外框，检查四边留白是否等宽",
            confidence=_confidence(rel / BBOX_TOL, area_ratio),
            metric={"edge": edge_name, "dev_px": int(round(dev)),
                    "dev_ratio": round(rel, 4)},
            region=bbox_region,
        ))

    # 3) 质心位移 -------------------------------------------------------
    dx = float(c["centroid"][0]) - float(r["centroid"][0])
    dy = float(c["centroid"][1]) - float(r["centroid"][1])
    dist = float(np.hypot(dx, dy))
    shift_ratio = dist / max(1.0, diag)
    if shift_ratio > SHIFT_TOL:
        if abs(dx) >= abs(dy):
            direction = "偏右" if dx > 0 else "偏左"
        else:
            direction = "偏下" if dy > 0 else "偏上"
        parts = []
        if abs(dx) >= 1.0:
            parts.append(f"偏{'右' if dx > 0 else '左'} {abs(dx):.0f}px")
        if abs(dy) >= 1.0:
            parts.append(f"偏{'下' if dy > 0 else '上'} {abs(dy):.0f}px")
        detail = "、".join(parts) if parts else "偏移"
        corr = ""
        if abs(dx) >= 1.0:
            corr += "左" if dx > 0 else "右"
        if abs(dy) >= 1.0:
            corr += "上" if dy > 0 else "下"
        sev = 25.0 + 60.0 * min(1.0, (shift_ratio - SHIFT_TOL) / 0.10)
        bbox_region = grid_cell(cx0 + cw / 2.0, cy0 + ch / 2.0, W, H)
        findings.append(Finding(
            code="PROP_SHIFT",
            layer="proportion",
            severity=sev,
            position=cell_label(*bbox_region),
            phenomenon=f"整体位置比样张{direction}，主体没有落在同一视觉中心",
            quantity=(f"质心{detail}（总位移 {dist:.0f}px，"
                      f"占对角线 {shift_ratio * 100:.1f}%）"),
            suggestion=(f"整体向{corr or '样张质心'}平移约 {dist:.0f}px，"
                        f"先对齐质心，再复查四边留白"),
            recheck="把两张图质心对齐后快速切换，看主体是否“跳”",
            confidence=_confidence(shift_ratio / SHIFT_TOL, area_ratio),
            metric={"dx_px": round(dx, 1), "dy_px": round(dy, 1),
                    "dist_px": round(dist, 1), "dist_ratio": round(shift_ratio, 4)},
            region=bbox_region,
        ))

    # 4) 主轴倾角 -------------------------------------------------------
    a_ref = float(r["angle_deg"])
    a_copy = float(c["angle_deg"])
    # 角度差归一到 (-90, 90]，避免 ±90 处跳变
    d_ang = ((a_copy - a_ref + 90.0) % 180.0) - 90.0
    if abs(d_ang) > ANGLE_TOL:
        cw_wise = d_ang > 0
        direction = "顺时针" if cw_wise else "逆时针"
        fix = "逆时针" if cw_wise else "顺时针"
        sev = 25.0 + 60.0 * min(1.0, (abs(d_ang) - ANGLE_TOL) / 20.0)
        findings.append(Finding(
            code="PROP_ANGLE",
            layer="proportion",
            severity=sev,
            position="整体构图",
            phenomenon=f"整体主轴比样张{direction}偏转",
            quantity=(f"主轴倾角 {a_copy:.1f}°，样张 {a_ref:.1f}°"
                      f"（相差 {abs(d_ang):.1f}°，{direction}偏转）"),
            suggestion=(f"以质心为轴整体向{fix}回转约 {abs(d_ang):.1f}°，"
                        f"先对齐主轴方向再修轮廓"),
            recheck="沿轮廓画一条贯穿长轴，和样张比对倾角",
            confidence=_confidence(abs(d_ang) / ANGLE_TOL, area_ratio),
            metric={"angle_copy": round(a_copy, 2), "angle_ref": round(a_ref, 2),
                    "angle_diff": round(d_ang, 2)},
            region=grid_cell(c["centroid"][0], c["centroid"][1], W, H),
        ))

    # 每类至多 1 条（上面天然满足）；按严重度取前 3。
    candidates = [f for f in findings if f.confidence >= min_conf]
    candidates.sort(key=lambda f: f.severity, reverse=True)
    return candidates[:MAX_FINDINGS]


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import re

    # 460 画布：对角线 650.5px，20px 位移 = 3.07% > 3%，刚好能测到 PROP_SHIFT。
    W = H = 460
    ref = np.zeros((H, W), np.uint8)
    cv2.circle(ref, (230, 120), 40, 255, -1)          # 圆头
    cv2.rectangle(ref, (30, 150), (429, 379), 255, -1)  # 矩形身体
    # 头部与身体连成一体；bbox = (30, 80, 400, 300)
    valid = np.full((H, W), 255, np.uint8)

    # copy：宽度压到 86%（绕 bbox 中心 x=229.5 缩放），整体下移 20px
    M = np.float32([[0.86, 0.0, (1.0 - 0.86) * 229.5], [0.0, 1.0, 20.0]])
    copy = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)

    findings = compare_proportion(ref, copy, valid)
    codes = [f.code for f in findings]
    assert len(findings) <= 3, f"总条数超过 3: {codes}"
    assert "PROP_ASPECT" in codes, f"未检出 PROP_ASPECT: {codes}"
    assert "PROP_SHIFT" in codes, f"未检出 PROP_SHIFT: {codes}"

    asp = next(f for f in findings if f.code == "PROP_ASPECT")
    decimals = [float(x) for x in re.findall(r"\d+\.\d+", asp.quantity)]
    assert any(abs(v - 0.86) < 0.02 for v in decimals), \
        f"长宽比比值应接近 0.86，实际 quantity={asp.quantity!r}"
    sh = next(f for f in findings if f.code == "PROP_SHIFT")
    assert "20px" in sh.quantity or "偏下" in sh.quantity, sh.quantity
    assert sh.region[0] >= 2, f"下移后质心应在下半区: {sh.region}"

    # 空输入不崩
    assert measure(np.zeros((50, 50), np.uint8)) == {"empty": True}
    assert compare_proportion(np.zeros((50, 50), np.uint8),
                             np.zeros((50, 50), np.uint8), None) == []
    assert measure(ref, np.zeros((H, W), np.uint8)) == {"empty": True}
    # 空 valid 时面积比的分母是整幅画布
    m = measure(ref)
    assert m["bbox"] == (30, 80, 400, 300), m["bbox"]
    assert abs(m["aspect"] - 400.0 / 300.0) < 0.02, m["aspect"]

    print("[proportion] PASS")
    print("  codes:", codes)
    for f in findings:
        print(f"  {f.code}: {f.quantity} | position={f.position} region={f.region}")
