"""参考图/临摹图配准。

约定: H 为 3x3 float64, 把 copy(临摹图) 坐标映射到 ref(参考图) 坐标;
warped/warped_gray/warped_fg/valid 均为 ref 画布尺寸。

公开 API:
    warp_to_ref(img, H, size_wh) -> (warped, valid)
    align_pair(ref, copy) -> AlignResult
    align_manual(ref, copy, quad_xy, order) -> AlignResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .io_prep import Prep

__all__ = ["AlignResult", "warp_to_ref", "align_pair", "align_manual"]

RATIO_TEST = 0.75
RANSAC_PX = 3.0
REFINE_SIDE = 600
OK_INLIER_RATIO = 0.15
OK_RESIDUAL_PX = 4.0
# 相似变换只有 4 自由度，画得"不太像"时残差天然比单应大，
# 阈值放宽到 8px；超过就说明连整体位置都对不上，才判失败。
OK_RESIDUAL_SIM_PX = 8.0
# 单应透视项强度（|h31|*W + |h32|*H）超过它，认为这张是斜拍的，比例结论要打折。
PERSPECTIVE_TOL = 0.06
OK_MIN_INLIERS = 10

_CORNER_ORDER = ["tl", "tr", "br", "bl"]


@dataclass
class AlignResult:
    """配准结果。"""
    ok: bool
    H: np.ndarray          # 3x3 float64, copy -> ref
    warped: np.ndarray     # copy 变换到 ref 画布的 BGR
    warped_gray: np.ndarray
    warped_fg: np.ndarray  # copy 前景 mask 变换后的 uint8 0/255
    valid: np.ndarray      # 有效区域 mask uint8 0/255
    inlier_ratio: float
    residual_px: float
    message: str
    model: str = "homography"   # similarity | homography | manual
    warn: str = ""              # 需要提醒用户但不算失败的情况


def warp_to_ref(img: np.ndarray, H: np.ndarray,
                size_wh: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """把 img 按 H 变换到 (w, h) 画布，并返回有效区域 mask。"""
    w = int(size_wh[0])
    h = int(size_wh[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"非法画布尺寸: {size_wh}")
    M = _as_h(H)
    warped = cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    src_valid = np.full(img.shape[:2], 255, dtype=np.uint8)
    moved = cv2.warpPerspective(
        src_valid, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = np.where(moved >= 250, 255, 0).astype(np.uint8)
    return warped, valid


def _as_h(H) -> np.ndarray:
    """规整为有限的 3x3 float64；不可用时退回 identity。"""
    try:
        a = np.asarray(H, dtype=np.float64)
    except (TypeError, ValueError):
        return np.eye(3, dtype=np.float64)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        return np.eye(3, dtype=np.float64)
    if abs(float(np.linalg.det(a))) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return a.copy()


def _make_result(ok: bool, H, ref: Prep, copy: Prep, inlier_ratio: float,
                 residual_px: float, message: str,
                 model: str = "homography", warn: str = "") -> AlignResult:
    H = _as_h(H)
    h, w = ref.bgr.shape[:2]
    warped, valid = warp_to_ref(copy.bgr, H, (w, h))
    warped_gray, _ = warp_to_ref(copy.gray, H, (w, h))
    fg_moved, _ = warp_to_ref(copy.fg, H, (w, h))
    warped_fg = np.where((fg_moved >= 128) & (valid > 0), 255, 0).astype(np.uint8)
    return AlignResult(
        ok=bool(ok), H=H, warped=warped, warped_gray=warped_gray,
        warped_fg=warped_fg, valid=valid,
        inlier_ratio=float(inlier_ratio),
        residual_px=float(residual_px), message=str(message), model=str(model),
        warn=str(warn))


def _good_matches(des_ref: np.ndarray, des_copy: np.ndarray) -> List[cv2.DMatch]:
    """BFMatcher(HAMMING) + knn k=2 + Lowe ratio。"""
    if des_ref is None or des_copy is None or len(des_ref) < 2 or len(des_copy) < 2:
        return []
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des_copy, des_ref, k=2)
    except cv2.error:
        return []
    good: List[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if float(first.distance) < RATIO_TEST * float(second.distance):
            good.append(first)
    return good


def _reproj_errors(H, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """逐点重投影误差（像素），输入不合法时返回全 inf。"""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) == 0 or len(src) != len(dst):
        return np.full(len(src), np.inf)
    try:
        proj = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), np.asarray(H, dtype=np.float64)).reshape(-1, 2)
    except cv2.error:
        return np.full(len(src), np.inf)
    return np.linalg.norm(proj - dst, axis=1)


def _reproj_median(H, src: np.ndarray, dst: np.ndarray,
                   inlier_mask: Optional[np.ndarray] = None) -> float:
    """投影后逐点误差的中位数（像素）。"""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) == 0 or len(src) != len(dst):
        return float("inf")
    try:
        proj = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), np.asarray(H, dtype=np.float64)).reshape(-1, 2)
    except cv2.error:
        return float("inf")
    err = np.linalg.norm(proj - dst, axis=1)
    err = err[np.isfinite(err)]
    if inlier_mask is not None:
        sel = err[np.asarray(inlier_mask, dtype=bool)]
        if sel.size:
            err = sel
    if err.size == 0:
        return float("inf")
    return float(np.median(err))


def _scale_matrix(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def _small_pair(ref_gray: np.ndarray, copy_gray: np.ndarray):
    """把两图缩到长边 REFINE_SIDE，返回 (ref_s, copy_s, S_copy, S_ref)。"""
    if ref_gray.size == 0 or copy_gray.size == 0:
        return None
    hr, wr = ref_gray.shape[:2]
    hc, wc = copy_gray.shape[:2]
    sr = min(1.0, REFINE_SIDE / float(max(hr, wr)))
    sc = min(1.0, REFINE_SIDE / float(max(hc, wc)))
    wr2 = max(1, int(round(wr * sr)))
    hr2 = max(1, int(round(hr * sr)))
    wc2 = max(1, int(round(wc * sc)))
    hc2 = max(1, int(round(hc * sc)))
    ref_s = ref_gray if (wr2, hr2) == (wr, hr) else cv2.resize(
        ref_gray, (wr2, hr2), interpolation=cv2.INTER_AREA)
    copy_s = copy_gray if (wc2, hc2) == (wc, hc) else cv2.resize(
        copy_gray, (wc2, hc2), interpolation=cv2.INTER_AREA)
    S_ref = _scale_matrix(wr2 / float(wr), hr2 / float(hr))
    S_copy = _scale_matrix(wc2 / float(wc), hc2 / float(hc))
    return (ref_s.astype(np.float32), copy_s.astype(np.float32),
            S_copy, S_ref)


def _match_score(ref_s: np.ndarray, copy_s: np.ndarray, W: np.ndarray) -> float:
    """把 copy_s 按 W 投到 ref 画布后，重叠区域的归一化相关系数。"""
    h, w = ref_s.shape[:2]
    try:
        warped = cv2.warpPerspective(
            copy_s, W, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(
            np.full(copy_s.shape, 255, dtype=np.uint8), W, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
    except cv2.error:
        return -2.0
    mask = valid > 127
    if int(mask.sum()) < int(0.2 * mask.size):
        return -2.0
    a = ref_s[mask].astype(np.float32)
    b = warped[mask].astype(np.float32)
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return -2.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else -2.0


def _ecc_homography(ref_s: np.ndarray, copy_s: np.ndarray,
                    H_init: np.ndarray) -> Optional[np.ndarray]:
    """ECC(MOTION_HOMOGRAM) 精修。OpenCV 的方向约定不直观，这里对结果与
    其逆各算一次相关性，取真正把 copy 投到 ref 的那个方向。"""
    # OpenCV 的常量名是 MOTION_HOMOGRAPHY；兼容旧写法 MOTION_HOMOGRAM。
    motion = getattr(cv2, "MOTION_HOMOGRAPHY", None)
    if motion is None:
        motion = getattr(cv2, "MOTION_HOMOGRAM", None)
    if motion is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    W = np.ascontiguousarray(np.asarray(H_init, dtype=np.float32))
    try:
        _, Wecc = cv2.findTransformECC(ref_s, copy_s, W, motion,
                                       criteria, None, 5)
    except (cv2.error, ValueError, TypeError):
        return None
    Wecc = np.asarray(Wecc, dtype=np.float64)
    if not np.all(np.isfinite(Wecc)) or abs(float(np.linalg.det(Wecc))) < 1e-8:
        return None
    candidates = [Wecc, np.linalg.inv(Wecc)]
    best: Optional[np.ndarray] = None
    best_score = -2.0
    for cand in candidates:
        score = _match_score(ref_s, copy_s, cand)
        if score > best_score:
            best, best_score = cand, score
    if best is None or best_score < 0.2:
        return None
    return best


def _refine_homography(ref_gray: np.ndarray, copy_gray: np.ndarray, H: np.ndarray,
                       src_pts: np.ndarray, dst_pts: np.ndarray):
    """ECC 精修，失败时退回内点仿射二次拟合；返回 (H_best, residual) 或 (None, inf)。"""
    try:
        small = _small_pair(ref_gray, copy_gray)
    except (cv2.error, ValueError):
        return None, float("inf")
    if small is None:
        return None, float("inf")
    ref_s, copy_s, S_copy, S_ref = small

    candidates: List[np.ndarray] = []
    try:
        H_small = S_ref @ H @ np.linalg.inv(S_copy)
    except np.linalg.LinAlgError:
        return None, float("inf")
    Wecc = _ecc_homography(ref_s, copy_s, H_small)
    if Wecc is not None:
        try:
            candidates.append(np.linalg.inv(S_ref) @ Wecc @ S_copy)
        except np.linalg.LinAlgError:
            pass

    try:
        M, _ = cv2.estimateAffinePartial2D(
            np.asarray(src_pts, np.float32).reshape(-1, 1, 2),
            np.asarray(dst_pts, np.float32).reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        if M is not None and np.all(np.isfinite(M)):
            candidates.append(np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64))
    except cv2.error:
        pass

    best_H: Optional[np.ndarray] = None
    best_err = float("inf")
    for cand in candidates:
        cand = _as_h(cand)
        if abs(float(np.linalg.det(cand[:2, :2]))) < 1e-3:
            continue
        err = _reproj_median(cand, src_pts, dst_pts)
        if math.isfinite(err) and err < best_err:
            best_H, best_err = cand, err
    if best_H is None:
        return None, float("inf")
    return best_H, best_err


def align_pair(ref: Prep, copy: Prep, model: str = "similarity") -> AlignResult:
    """ORB+RANSAC 自动配准。

    ``model="similarity"``（默认）只用 旋转+等比缩放+平移 4 自由度对齐：
    这样"画窄了/画矮了"之类真实的比例错误不会被对齐吃掉（单应会把它们
    吸收成透视，看起来就"没错"了）。证件照式的斜拍纸张才需要用
    ``model="homography"``。
    """
    identity = np.eye(3, dtype=np.float64)
    if ref.gray.size == 0 or copy.gray.size == 0:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "图像为空，无法配准", model=model)
    try:
        orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8)
        kp_ref, des_ref = orb.detectAndCompute(ref.gray, None)
        kp_copy, des_copy = orb.detectAndCompute(copy.gray, None)
    except cv2.error as exc:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            f"特征提取失败：{exc}", model=model)
    n_ref = 0 if kp_ref is None else len(kp_ref)
    n_copy = 0 if kp_copy is None else len(kp_copy)
    if des_ref is None or des_copy is None or n_ref < 8 or n_copy < 8:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            f"特征点不足（参考 {n_ref}，临摹 {n_copy}）", model=model)

    good = _good_matches(des_ref, des_copy)
    if len(good) < 6:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            f"特征点不足（好匹配 {len(good)} < 6）", model=model)

    src = np.float64([[kp_copy[m.queryIdx].pt] for m in good]).reshape(-1, 2)
    dst = np.float64([[kp_ref[m.trainIdx].pt] for m in good]).reshape(-1, 2)
    try:
        H_h, mask_h = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                         cv2.RANSAC, RANSAC_PX)
    except cv2.error:
        H_h, mask_h = None, None
    if H_h is None or not np.all(np.isfinite(H_h)):
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "单应估计失败（匹配点退化）", model=model)

    H_h = _as_h(H_h)
    inl_h = (np.ones(len(good), dtype=bool) if mask_h is None
             else np.asarray(mask_h, dtype=bool).ravel())
    residual_h = _reproj_median(H_h, src, dst, inl_h)
    if int(inl_h.sum()) >= 12 and residual_h > 2.5:
        H2, residual2 = _refine_homography(
            ref.gray, copy.gray, H_h, src[inl_h], dst[inl_h])
        if (H2 is not None and math.isfinite(residual2) and residual2 < residual_h):
            H_h, residual_h = H2, residual2

    H_s, inl_s, residual_s = None, np.zeros_like(inl_h), float("inf")
    if model == "similarity":
        try:
            M, ms = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                maxIters=5000, confidence=0.99, refineIters=20)
        except cv2.error:
            M, ms = None, None
        if M is not None and np.all(np.isfinite(M)):
            H_s = np.vstack([np.asarray(M, dtype=np.float64), [0.0, 0.0, 1.0]])
            inl_s = (np.ones(len(good), dtype=bool) if ms is None
                     else np.asarray(ms, dtype=bool).ravel())
            residual_s = _reproj_median(H_s, src, dst, inl_s)

    use_sim = (model == "similarity" and H_s is not None
               and int(inl_s.sum()) >= OK_MIN_INLIERS)
    warn = ""
    if use_sim:
        H = H_s
        inliers = inl_s
        residual = residual_s
        chosen = "similarity"
        ok_residual = OK_RESIDUAL_SIM_PX
        note = f"（相似变换；单应残差 {residual_h:.2f}px 仅供参考）"
    else:
        H = H_h
        inliers = inl_h
        residual = residual_h
        chosen = "homography"
        ok_residual = OK_RESIDUAL_PX
        note = ""
        if model == "homography":
            warn = ("⚠ 已用单应（允许透视）配准：真画歪的透视也会被对齐吃掉，"
                    "比例类结论只能当参考；若是扫描/正面平拍，建议改用 --align similarity。")

    if use_sim and H_h is not None and int(inl_h.sum()) >= OK_MIN_INLIERS:
        # 「该用单应还是相似变换」不能比两者的残差（各自只在自己的内点上算，
        # 天然都很小）。也不能只看"有多少点只有透视能解释"——把人物画窄，
        # 在特征点看来和把纸拍斜很像。可靠的区别是单应里真正的透视项有多强：
        # 纯缩放/旋转的 h31、h32 约为 0；真透视明显非零。
        w_img, h_img = float(copy.bgr.shape[1]), float(copy.bgr.shape[0])
        persp = abs(H_h[2, 0]) * w_img + abs(H_h[2, 1]) * h_img
        if persp > PERSPECTIVE_TOL:
            warn = (f"⚠ 临摹里检测到透视分量（{persp:.2f}）：如果你是用手机斜着拍的，"
                    f"比例类结论可能把拍摄变形算成「画错了」，请平拍或用 "
                    f"--align homography 复核；若是扫描/正面平拍，按下面的比例结论改就行。")

    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / float(len(good))

    ok = (inlier_ratio >= OK_INLIER_RATIO
          and residual <= ok_residual
          and n_inliers >= OK_MIN_INLIERS)
    if ok:
        message = (f"对齐成功（{chosen}，内点 {n_inliers}/{len(good)}，"
                   f"残差 {residual:.2f}px）{note}")
    else:
        reasons = []
        if n_inliers < OK_MIN_INLIERS:
            reasons.append(f"内点不足（{n_inliers}）")
        if inlier_ratio < OK_INLIER_RATIO:
            reasons.append(f"内点占比低（{inlier_ratio:.2f}）")
        if residual > ok_residual:
            reasons.append(f"残差过大（{residual:.2f}px）")
        message = f"对齐失败（{chosen}）：" + "；".join(reasons)
    return _make_result(ok, H, ref, copy, inlier_ratio, residual, message,
                        model=chosen, warn=warn)


def _parse_order(order) -> List[str]:
    """把 order 解析成「输入点顺序 -> 角点名」的列表。"""
    if isinstance(order, str):
        items = [part.strip().lower() for part in order.split(",") if part.strip()]
    else:
        try:
            items = [str(part).strip().lower() for part in order]
        except TypeError:
            items = []
    if len(items) == 4 and set(items) == set(_CORNER_ORDER):
        return items
    return list(_CORNER_ORDER)


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def align_manual(ref: Prep, copy: Prep, quad_xy: Sequence[Sequence[float]],
                 order: str = "tl,tr,br,bl") -> AlignResult:
    """用 copy 上的 4 个点对齐到 ref 画布四角。"""
    h, w = ref.bgr.shape[:2]
    identity = np.eye(3, dtype=np.float64)
    try:
        pts = np.asarray(quad_xy, dtype=np.float64)
    except (TypeError, ValueError):
        pts = np.empty((0, 0))
    if pts.size == 8:
        pts = pts.reshape(4, 2)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "手动四点无效：需要 4 个有限二维坐标")

    labels = _parse_order(order)
    src = np.array([pts[labels.index(name)] for name in _CORNER_ORDER],
                   dtype=np.float64)
    dst = np.array([[0.0, 0.0], [w - 1.0, 0.0],
                    [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
    if _polygon_area(src) < 1.0:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "手动四点退化：面积过小或点重合")
    try:
        H = cv2.getPerspectiveTransform(src.astype(np.float32),
                                        dst.astype(np.float32))
    except cv2.error:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "手动四点退化：无法求解单应")
    H = np.asarray(H, dtype=np.float64)
    if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
        return _make_result(False, identity, ref, copy, 0.0, float("inf"),
                            "手动四点退化：单应不可逆")
    return _make_result(True, H, ref, copy, 1.0, 0.0, "手动四点对齐完成")


def _demo_scene(w: int = 800, h: int = 600) -> np.ndarray:
    """自检用合成图：多组几何图元 + 文字，保证 ORB 有足够特征。"""
    rng = np.random.default_rng(11)
    img = np.full((h, w, 3), 244, dtype=np.uint8)
    for i in range(45):
        p1 = (int(rng.integers(25, w - 25)), int(rng.integers(25, h - 25)))
        c = int(rng.integers(10, 110))
        if i % 3 == 0:
            cv2.circle(img, p1, int(rng.integers(10, 70)), (c, c, c), 2, cv2.LINE_AA)
        elif i % 3 == 1:
            p2 = (int(rng.integers(25, w - 25)), int(rng.integers(25, h - 25)))
            cv2.line(img, p1, p2, (c, c, c), 2, cv2.LINE_AA)
        else:
            p2 = (int(rng.integers(25, w - 25)), int(rng.integers(25, h - 25)))
            cv2.rectangle(img, p1, p2, (c, c, c), 2, cv2.LINE_AA)
    for i, text in enumerate(("REF 001", "diag 45", "value 30", "edge test")):
        cv2.putText(img, text, (45, 110 + i * 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def _self_test() -> None:
    from .io_prep import prep_from_bgr

    w, h = 800, 600
    ref_bgr = _demo_scene(w, h)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.0, 1.0)
    M[0, 2] += 6.0
    M[1, 2] += 6.0
    copy_bgr = cv2.warpAffine(ref_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    copy_bgr = np.clip(copy_bgr.astype(np.float32) * 1.05 + 6.0, 0, 255
                       ).astype(np.uint8)

    ref = prep_from_bgr(ref_bgr)
    copy = prep_from_bgr(copy_bgr)

    res = align_pair(ref, copy)
    assert res.ok, f"align_pair 未通过: {res.message}"
    assert res.residual_px < 1.5, f"residual_px={res.residual_px:.3f} 超阈值"
    assert res.H.shape == (3, 3) and res.H.dtype == np.float64
    assert res.warped.shape[:2] == (h, w)
    assert res.warped_gray.shape == (h, w)
    assert set(np.unique(res.valid).tolist()) <= {0, 255}
    assert set(np.unique(res.warped_fg).tolist()) <= {0, 255}
    assert 0.0 <= res.inlier_ratio <= 1.0

    # 手动四点：把 ref 四角用真值逆变换映射回 copy，应还原同一 H。
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    ref_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                           dtype=np.float64).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(ref_corners, np.linalg.inv(M3)).reshape(-1, 2)
    man = align_manual(ref, copy, quad)
    assert man.ok, man.message
    proj = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), man.H).reshape(-1, 2)
    man_err = float(np.median(np.linalg.norm(
        proj - ref_corners.reshape(-1, 2), axis=1)))
    assert man_err < 1e-3, man_err
    assert not align_manual(ref, copy, [[0, 0], [0, 0], [0, 0], [0, 0]]).ok

    print(f"align PASS residual={res.residual_px:.3f}px "
          f"inlier_ratio={res.inlier_ratio:.2f} manual_err={man_err:.4f}px")


if __name__ == "__main__":
    _self_test()
