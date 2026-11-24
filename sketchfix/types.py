"""共享类型与常量。所有子模块只依赖本文件，避免循环导入。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import cv2
import numpy as np

# ---- 全局约定 ----
# 所有图像: BGR, uint8, 已摆正、长边 <= MAX_SIDE
# 所有 mask: uint8, 0 或 255, 255 = 前景/墨迹
MAX_SIDE = 1400
GRID_ROWS = 4
GRID_COLS = 4

LAYER_NAMES = {
    "proportion": "构图/比例",
    "line": "线稿/轮廓",
    "value": "明暗关系",
}

_ROW_LABEL = ["上", "中上", "中下", "下"]
_COL_LABEL = ["左", "中左", "中右", "右"]


def cell_label(r: int, c: int) -> str:
    """网格单元的人类可读位置名。"""
    if r < 0 or c < 0:
        return "整幅"
    row = _ROW_LABEL[min(r, 3)] if GRID_ROWS == 4 else f"第{r + 1}行"
    col = _COL_LABEL[min(c, 3)] if GRID_COLS == 4 else f"第{c + 1}列"
    return f"{row}{col}区"


def grid_cell(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    """把像素坐标映射到网格 (row, col)，越界则夹紧。"""
    if w <= 0 or h <= 0:
        return -1, -1
    c = int(np.clip(x / w * GRID_COLS, 0, GRID_COLS - 1))
    r = int(np.clip(y / h * GRID_ROWS, 0, GRID_ROWS - 1))
    return r, c


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def odd_kernel(k: int, minimum: int = 3) -> int:
    k = max(int(k), minimum)
    return k if k % 2 == 1 else k + 1


def clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def resize_long_side(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


@dataclass
class Finding:
    """一条可执行建议。位置+现象+量化偏差+改法+复查+置信度 六项齐全。"""
    code: str                 # 稳定机器码, 如 PROP_ASPECT / LINE_MISSING / VAL_DARK
    layer: str                # proportion | line | value
    severity: float           # 0-100, 越大越严重
    position: str             # 人类可读位置, 如 "上左区"
    phenomenon: str           # 现象一句话
    quantity: str             # 量化偏差, 带数字和单位
    suggestion: str           # 改法
    recheck: str              # 复查方法
    confidence: float = 0.5   # 0-1
    metric: Dict[str, Any] = field(default_factory=dict)
    region: Tuple[int, int] = (-1, -1)

    def __post_init__(self) -> None:
        self.confidence = clamp01(self.confidence)
        self.severity = float(np.clip(self.severity, 0.0, 100.0))
        if not self.position and self.region[0] >= 0:
            self.position = cell_label(*self.region)

    @property
    def layer_name(self) -> str:
        return LAYER_NAMES.get(self.layer, self.layer)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "layer": self.layer,
            "layer_name": self.layer_name,
            "severity": round(self.severity, 1),
            "position": self.position,
            "region": list(self.region),
            "phenomenon": self.phenomenon,
            "quantity": self.quantity,
            "suggestion": self.suggestion,
            "recheck": self.recheck,
            "confidence": round(self.confidence, 2),
        }
        d.update(self.metric)
        return d


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
