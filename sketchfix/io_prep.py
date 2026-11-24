"""图像读写、EXIF 摆正、缩放、光照校正与墨迹前景提取。

公开 API:
    imread_unicode(path) -> np.ndarray    # 支持中文路径的 BGR 读取
    imwrite_unicode(path, img) -> bool    # 支持中文路径的写出
    load_prep(path, max_side) -> Prep     # 从磁盘加载并预处理
    prep_from_bgr(bgr, max_side, path) -> Prep  # 从内存 BGR 预处理
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .types import MAX_SIDE, resize_long_side, to_gray

__all__ = [
    "Prep",
    "imread_unicode",
    "imwrite_unicode",
    "load_prep",
    "prep_from_bgr",
]


@dataclass
class Prep:
    """预处理结果。"""
    bgr: np.ndarray        # EXIF 摆正 + 长边缩放后的 BGR
    gray: np.ndarray       # 光照/纸张底色校正后的灰度 uint8
    raw_gray: np.ndarray   # 未校正灰度
    fg: np.ndarray         # 前景(墨迹) mask uint8 0/255
    scale: float           # bgr 相对原始(摆正后)图像的缩放比
    path: str = ""


def imread_unicode(path: str) -> np.ndarray:
    """读取图像为 BGR uint8；路径含中文时 cv2.imread 会失败，故用 fromfile+imdecode。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise FileNotFoundError(f"无法读取图像文件: {path}") from exc
    if buf.size == 0:
        raise ValueError(f"图像文件为空: {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法解码图像: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """写出图像；路径含中文时使用 imencode + tofile。失败返回 False 而不抛异常。"""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return False
    ext = os.path.splitext(str(path))[1]
    if not ext:
        ext = ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        buf.tofile(path)
        return True
    except (OSError, cv2.error, ValueError, TypeError):
        return False


def _read_bgr_exif(path: str) -> np.ndarray:
    """PIL 处理 EXIF 方向后转 BGR；PIL 不支持的格式退回 imdecode。"""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            arr = np.array(im)
    except (OSError, ValueError, Image.DecompressionBombError):
        return imread_unicode(path)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)


def _normalize_gray(gray: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    """大核高斯背景估计 + 逐像素相除 + 2%~98% 分位线性拉伸。

    背景 sigma 取长边的 1/3（而不是 1/12）：只压掉拍摄造成的**整体**明暗渐变，
    不把"某一大块画脏/画亮"这种真实偏差一并当光照除掉——否则明暗层会失明。

    返回 (校正灰度, 是否近似无对比度, 残差)。残差 = 背景 - 原图，墨迹与
    局部偏暗处为正，供前景提取使用。无对比度时给出常量灰度且不产生前景，
    保证纯白/纯黑图有确定结果。
    """
    gf = gray.astype(np.float32)
    long_side = max(gray.shape[:2])
    sigma = max(40.0, long_side / 3.0)
    bg = cv2.GaussianBlur(gf, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = np.maximum(bg, 1.0)
    resid = bg - gf
    # 固定锚点：把纸白压到 235 左右，而不是按本图分位做拉伸。
    # 逐图拉伸会让"整幅偏暗"被拉回正常，且两张图的增益不同，明暗层直接失去可比性。
    norm = cv2.divide(gf, bg, scale=235.0)

    lo, hi = (float(v) for v in np.percentile(norm, (2.0, 98.0)))
    if hi - lo < 4.0:
        level = 235.0 if float(norm.mean()) >= 120.0 else 0.0
        out = np.full(norm.shape, level, dtype=np.float32)
        return out.astype(np.uint8), True, np.zeros_like(gf)
    out = np.clip(norm, 0.0, 255.0)
    return out.astype(np.uint8), False, resid


def _ink_mask(gray: np.ndarray, low_contrast: bool, resid: np.ndarray) -> np.ndarray:
    """残差 Otsu 取墨迹 -> 3x3 close -> 去碎点 -> 3x3 open。

    直接对整幅灰度做 Otsu 在"纸张带渐变"时会切掉半张纸（前景占比 80%+），
    因此改为对"背景-原图"的残差做 Otsu，并加占比护栏。
    """
    if low_contrast:
        return np.zeros_like(gray, dtype=np.uint8)
    total = int(gray.shape[0]) * int(gray.shape[1])
    r8 = np.clip(resid, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(r8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = max(float(t), 12.0)
    # 护栏：墨迹不该占画面一半；占比过高说明阈值切到了纸，逐步抬高阈值。
    fg = (resid >= t).astype(np.uint8) * 255
    while fg.mean() > 0.45 * 255 and t < 110.0:
        t += 8.0
        fg = (resid >= t).astype(np.uint8) * 255
    if fg.mean() == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray, dtype=np.uint8)
    min_area = max(4, int(round(0.0002 * total)))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    fg = np.where(keep[labels], 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    return fg


def prep_from_bgr(bgr: np.ndarray, max_side: int = MAX_SIDE,
                  path: str = "") -> Prep:
    """由内存 BGR 图像生成 Prep（EXIF 已由调用方负责）。"""
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
        raise ValueError("输入图像为空")
    img = bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"不支持的图像形状: {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    orig_long = max(int(img.shape[0]), int(img.shape[1]))
    scale = 1.0
    if max_side and orig_long > int(max_side):
        img = resize_long_side(img, int(max_side))
        scale = max(int(img.shape[0]), int(img.shape[1])) / float(orig_long)

    raw_gray = to_gray(img)
    gray, low_contrast, resid = _normalize_gray(raw_gray)
    fg = _ink_mask(gray, low_contrast, resid)
    return Prep(bgr=img, gray=gray, raw_gray=raw_gray, fg=fg,
                scale=float(scale), path=str(path))


def load_prep(path: str, max_side: int = MAX_SIDE) -> Prep:
    """读图（EXIF 摆正）-> 缩放 -> 光照校正 -> 前景 mask。"""
    bgr = _read_bgr_exif(path)
    return prep_from_bgr(bgr, max_side=max_side, path=path)


def _demo_image(w: int = 800, h: int = 600) -> np.ndarray:
    """自检用合成图：渐变纸张底色 + 圆/线/矩形 + 文字特征。"""
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 246, dtype=np.uint8)
    grad = np.linspace(0.0, 16.0, w, dtype=np.float32)
    img = np.clip(img.astype(np.float32) - grad[None, :, None], 0, 255).astype(np.uint8)
    for i in range(30):
        p1 = (int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30)))
        c = int(rng.integers(15, 100))
        if i % 3 == 0:
            cv2.circle(img, p1, int(rng.integers(10, 60)), (c, c, c), 2, cv2.LINE_AA)
        elif i % 3 == 1:
            p2 = (int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30)))
            cv2.line(img, p1, p2, (c, c, c), 2, cv2.LINE_AA)
        else:
            p2 = (int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30)))
            cv2.rectangle(img, p1, p2, (c, c, c), 2, cv2.LINE_AA)
    for i, text in enumerate(("Sketch A", "line 01", "value 35", "perspective")):
        cv2.putText(img, text, (40, 100 + i * 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (30, 30, 30), 2, cv2.LINE_AA)
    return img


def _self_test() -> None:
    import shutil
    import tempfile

    img = _demo_image()
    tmp = tempfile.mkdtemp(prefix="sketchfix_io_")
    try:
        path = os.path.join(tmp, "测试_图像.png")
        assert imwrite_unicode(path, img), "imwrite_unicode 失败"
        assert os.path.isfile(path), "文件未落盘"
        back = imread_unicode(path)
        assert back.shape == img.shape and back.dtype == np.uint8
        assert int(np.abs(back.astype(np.int16) - img.astype(np.int16)).max()) <= 2

        prep = load_prep(path)
        h, w = prep.bgr.shape[:2]
        assert max(h, w) <= MAX_SIDE
        assert prep.gray.shape == (h, w) and prep.raw_gray.shape == (h, w)
        assert prep.fg.shape == (h, w) and set(np.unique(prep.fg)) <= {0, 255}
        fg_ratio = float((prep.fg > 0).mean())
        assert 0.003 < fg_ratio < 0.6, f"前景占比异常: {fg_ratio}"
        assert prep.scale > 0.0 and prep.path == path

        for level in (255, 0):
            flat = np.full((64, 80, 3), level, dtype=np.uint8)
            flatp = prep_from_bgr(flat)
            assert flatp.gray.shape == (64, 80)
            assert int(flatp.fg.max()) == 0, "纯色图不应有前景"

        big = prep_from_bgr(np.full((3000, 2000, 3), 255, np.uint8), max_side=MAX_SIDE)
        assert max(big.bgr.shape[:2]) == MAX_SIDE and big.scale < 1.0

        print(f"io_prep PASS fg_ratio={fg_ratio:.3f} scale={prep.scale:.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
