"""报告与叠加图生成。"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .types import Finding, GRID_COLS, GRID_ROWS, LAYER_NAMES, MAX_SIDE, ensure_dir


def overlay(ref_bgr: np.ndarray, findings: Sequence[Finding]) -> np.ndarray:
    """在样张上画编号标记 + 指向框（编号用 ASCII，避免 OpenCV 画不出中文）。

    框取**整格**：只框半格会让人以为问题只影响半格。编号圆点钉在框的左上角，
    并夹在画布内——第一列/第一行的框原来会把圆点画到画布外，看不见编号。
    """
    img = ref_bgr.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    colors = {"proportion": (60, 60, 230), "line": (230, 140, 40), "value": (40, 180, 60)}
    rad = 18
    for i, f in enumerate(findings, start=1):
        r, c = f.region
        col = colors.get(f.layer, (0, 0, 255))
        if r >= 0 and c >= 0:
            # 合并过的结论（"下左区+下中左区"）要把涉及的格子全框出来，
            # 否则文字说了两格、图上只标一格。
            cells = f.metric.get("merged_regions") or [[r, c]]
            cells = [(int(a), int(b)) for a, b in cells if int(a) >= 0 and int(b) >= 0]
            for (rr, cc) in cells:
                x0 = int(cc * w / GRID_COLS)
                y0 = int(rr * h / GRID_ROWS)
                x1 = int((cc + 1) * w / GRID_COLS)
                y1 = int((rr + 1) * h / GRID_ROWS)
                cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), col, 3)
            x0 = int(cells[0][1] * w / GRID_COLS)
            y0 = int(cells[0][0] * h / GRID_ROWS)
            bx = min(max(x0 + rad + 2, rad), w - rad)
            by = min(max(y0 + rad + 2, rad), h - rad)
            cv2.circle(img, (bx, by), rad, col, -1)
            label = str(i)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(img, label, (bx - tw // 2, by + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # 整幅性问题（比例/空白）没有格子可钉，编号列在左上角。
            cv2.putText(img, str(i), (20, 40 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    return img


def heatmap(ref_bgr: np.ndarray, findings: Sequence[Finding], grid: np.ndarray | None = None) -> np.ndarray:
    """粗粒度差异热图：4x4 格，颜色按严重度绝对值（不按本图最大值归一）。

    按图内最大值归一会把"唯一一条轻微问题"也画成满格红，等于夸大。
    这里固定用 0~100 的严重度刻度，并且把网格线画出来，方便对格子找位置。
    """
    img = ref_bgr if ref_bgr.ndim == 3 else cv2.cvtColor(ref_bgr, cv2.COLOR_GRAY2BGR)
    if grid is None:
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        for f in findings:
            # 合并结论（"下左区+下中左区"）涉及的格子都要着色
            cells = f.metric.get("merged_regions") or ([list(f.region)] if f.region[0] >= 0 else [])
            for r, c in cells:
                r, c = int(r), int(c)
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    grid[r, c] = max(grid[r, c], f.severity)
    g = np.asarray(grid, dtype=np.float32)
    if g.size == 0 or not np.isfinite(g).any():
        return img.copy()
    g = np.nan_to_num(g, nan=0.0)
    h, w = img.shape[:2]
    # 最近邻：格内颜色必须平坦，线性插值会把相邻格的高分"渗"过来，
    # 看起来像整排都红。
    m = cv2.resize(np.clip(g / 100.0, 0, 1), (w, h), interpolation=cv2.INTER_NEAREST)
    m8 = np.uint8(m * 255)
    cm = cv2.applyColorMap(m8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.62, cm, 0.38, 0)
    for i in range(1, GRID_COLS):
        x = int(i * w / GRID_COLS)
        cv2.line(out, (x, 0), (x, h), (170, 170, 170), 1, cv2.LINE_AA)
    for i in range(1, GRID_ROWS):
        y = int(i * h / GRID_ROWS)
        cv2.line(out, (0, y), (w, y), (170, 170, 170), 1, cv2.LINE_AA)
    # 有分的格子写清严重度，避免"看着都红"却分不出轻重
    for r in range(min(GRID_ROWS, g.shape[0])):
        for c in range(min(GRID_COLS, g.shape[1])):
            if g[r, c] <= 0:
                continue
            txt = f"{g[r, c]:.0f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = int((c + 1) * w / GRID_COLS) - tw - 8
            y = int(r * h / GRID_ROWS) + th + 8
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (30, 30, 30), 1, cv2.LINE_AA)
    return out


def _score(findings: Sequence[Finding]) -> float:
    """粗打分，只用来给"现在大概什么水平"一个锚点，不追求精确。

    只按严重度均值扣分会让"3 条问题"直接掉到 50 分以下，读起来像不及格，
    而实际只是"有 3 处要改"。所以：均摊系数降到 0.62，且超过 3 条时
    每条再补一点，避免条数无限多时分数虚高。
    """
    if not findings:
        return 92.0
    mean = sum(f.severity * f.confidence for f in findings) / max(1, len(findings))
    penalty = 0.62 * mean + 1.5 * max(0, len(findings) - 3)
    return float(np.clip(100.0 - penalty, 20.0, 99.0))


def _block(i: int, f: Finding) -> List[str]:
    return [
        f"### {i}.【{_sev_word(f.severity)}】{f.layer_name} · {f.position}",
        "",
        f"- 现象：{f.phenomenon}",
        f"- 量化：{f.quantity}",
        f"- 改法：{f.suggestion}",
        f"- 复查：{f.recheck}",
        f"- 置信度：{f.confidence:.2f}（{_conf_word(f.confidence)}）",
        "",
    ]


def _sev_word(s: float) -> str:
    return "明显" if s >= 70 else ("中等" if s >= 50 else "轻微")


def _conf_word(c: float) -> str:
    return "高" if c >= 0.7 else ("中" if c >= 0.45 else "低，建议人工复核")


def _score_word(s: float) -> str:
    """分数只是一个锚点，所以配一句人话，避免"58 分"被当成考试不及格。"""
    if s >= 85:
        return "整体接近样张，只剩细节要修"
    if s >= 70:
        return "大形已对，有几处要改"
    if s >= 50:
        return "有问题，先改排在前面的几条"
    return "偏差较大，建议先对形再深入"


def build_report(ref_path: str, copy_path: str, align_info: Dict,
                 findings: Sequence[Finding], notes: List[str], out_dir: str,
                 ref_bgr: np.ndarray | None = None, grid: np.ndarray | None = None) -> Dict[str, str]:
    ensure_dir(out_dir)
    paths = {
        "report": os.path.join(out_dir, "report.md"),
        "overlay": os.path.join(out_dir, "overlay.png"),
        "heatmap": os.path.join(out_dir, "heatmap.png"),
        "json": os.path.join(out_dir, "findings.json"),
    }
    if ref_bgr is not None:
        cv2.imencode(".png", overlay(ref_bgr, findings))[1].tofile(paths["overlay"])
        cv2.imencode(".png", heatmap(ref_bgr, findings, grid))[1].tofile(paths["heatmap"])

    score = _score(findings)
    if align_info.get("ok"):
        score_line = f"- 总评：**{score:.0f}/100**（{_score_word(score)}）"
    else:
        score_line = "- 总评：**未评分**（配准未通过，分数会误导，不作为依据）"
    lines: List[str] = [
        "# 临摹对比诊断报告",
        "",
        f"- 样张：`{os.path.basename(ref_path)}`",
        f"- 临摹：`{os.path.basename(copy_path)}`",
        f"- 配准：{'通过' if align_info.get('ok') else '未通过'}"
        f"（{align_info.get('model', 'homography')}）；"
        f"内点率 {align_info.get('inlier_ratio', 0):.2f}；"
        f"残差 {align_info.get('residual_px', float('nan')):.2f}px",
        score_line,
        "",
        "## 主要问题总述",
        "",
    ]
    if findings:
        top = "；".join(
            f"{f.phenomenon.rstrip('。')}（{f.position}）" for f in findings[:3])
        lines.append(f"共 {len(findings)} 条，按严重度排序。最突出：{top}。")
    else:
        lines.append("未发现超过阈值的偏差（或证据不足，见下方说明）。")
    lines.append("")

    if notes:
        lines += ["## 说明与限制", ""] + [f"- {n}" for n in notes] + [""]

    lines += ["## 问题清单", ""]
    if not findings:
        lines += ["（无）", ""]
    for i, f in enumerate(findings, start=1):
        lines += _block(i, f)

    lines += ["## 图", ""]
    if ref_bgr is not None:
        lines += [f"![编号标记](overlay.png)", "", f"![差异热图](heatmap.png)", ""]

    with open(paths["report"], "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump({"align": align_info, "notes": notes, "score": score,
                   "findings": [f.to_dict() for f in findings]},
                  fh, ensure_ascii=False, indent=2)
    return paths
