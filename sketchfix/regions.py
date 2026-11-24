"""分区聚合、排序与"诚实降级"判断。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .types import Finding, GRID_COLS, GRID_ROWS
from .types import cell_label


def grid_coverage(valid: np.ndarray, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> np.ndarray:
    """每格有效像素占比，float 0-1。"""
    h, w = valid.shape[:2]
    out = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * h / rows), int((r + 1) * h / rows)
            x0, x1 = int(c * w / cols), int((c + 1) * w / cols)
            blk = valid[y0:y1, x0:x1]
            out[r, c] = float((blk > 127).mean()) if blk.size else 0.0
    return out


def merge_adjacent(findings: Sequence[Finding]) -> List[Finding]:
    """同 code 且网格四邻相接的 findings 合并为一个，位置写成"A+B区"。"""
    kept: List[Finding] = []
    used = [False] * len(findings)
    order = sorted(range(len(findings)), key=lambda i: -findings[i].severity)
    for i in order:
        if used[i]:
            continue
        group = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(findings)):
                if used[j] or findings[j].code != findings[i].code:
                    continue
                for g in group:
                    if _adjacent(findings[g].region, findings[j].region):
                        group.append(j)
                        used[j] = True
                        changed = True
                        break
        base = findings[i]
        if len(group) == 1:
            kept.append(base)
            continue
        labels = [cell_label(*findings[g].region) for g in group if findings[g].region[0] >= 0]
        base.position = "+".join(dict.fromkeys(labels)) if labels else base.position
        base.severity = max(findings[g].severity for g in group)
        base.confidence = max(findings[g].confidence for g in group)
        base.metric["merged_regions"] = [list(findings[g].region) for g in group]
        kept.append(base)
    return kept


def _adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    if a[0] < 0 or b[0] < 0:
        return False
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def rank(findings: Sequence[Finding], top_n: int = 5, strictness: float = 1.0,
         min_conf: float = 0.30) -> List[Finding]:
    """按严重度排序、置信度过滤，返回前 top_n 条。

    strictness > 1 更严格（严重度阈值抬高），< 1 更宽松。
    整体比例类结论永远排在前面：它是"根因"，局部线稿问题多半由它派生。
    """
    pool = [f for f in findings if f.confidence >= min_conf]
    pool = [f for f in pool if f.severity * strictness >= RANK_MIN_SEVERITY]
    pool.sort(key=lambda f: (0 if f.code in GLOBAL_CODES else 1,
                             -f.severity, -f.confidence))
    return pool[:top_n]


# 整体形状类结论（根因层）。它们错了，局部逐区结论的可信度会整体下降。
GLOBAL_CODES = ("PROP_ASPECT", "PROP_BBOX")
# 进报告的最低严重度。阈值卡在 35 时，"刚好超过检出线"的问题（例如压暗
# 13 级 → 严重度 34.5）会被整条吞掉，等于检出了却不说。
RANK_MIN_SEVERITY = 30.0
# 完全由整体比例错误派生的结论：轮廓被拉伸后，逐区"线平移了"必然同时在
# 各个方向报出来，单独去挪某一块只会越改越乱。
DERIVED_CODES = ("LINE_SHIFT",)


def suppress_derived(findings: Sequence[Finding],
                     min_conf: float = 0.50) -> Tuple[List[Finding], List[str]]:
    """整体比例明显错时，压掉被它派生的逐区结论，并给出说明。

    返回 (保留的 findings, 说明文字列表)。全部输入照原样返回时说明为空。
    """
    glob = [f for f in findings
            if f.code in GLOBAL_CODES and f.confidence >= 0.6]
    if not glob:
        return list(findings), []
    keep: List[Finding] = []
    dropped: List[Finding] = []
    for f in findings:
        derived = (f.code in DERIVED_CODES
                   or (f.code in ("LINE_MISSING", "LINE_EXTRA")
                       and f.confidence < min_conf))
        if derived:
            dropped.append(f)
        else:
            keep.append(f)
    notes = []
    if dropped:
        notes.append(f"⚠ 整体比例偏差较大（{glob[0].quantity}），"
                     f"暂不列出 {len(dropped)} 条逐区线稿偏移/缺线结论："
                     f"先把整体比例改对，再看局部，否则会越改越乱。")
    return keep, notes


def blank_copy_finding(ref_fg: np.ndarray, copy_fg: np.ndarray,
                       min_ref_ink: int = 400, ratio: float = 0.15):
    """临摹几乎空白时给一条"整幅"结论；正常返回 None。

    空白临摹用逐区阈值去比毫无意义（每个区都"缺线"），先说清楚
    "你几乎什么都没画"更有用。
    """
    ref_ink = int((np.asarray(ref_fg) > 127).sum())
    cp_ink = int((np.asarray(copy_fg) > 127).sum())
    if ref_ink < min_ref_ink or cp_ink >= ratio * ref_ink:
        return None
    pct = cp_ink / max(1, ref_ink) * 100.0
    return Finding(
        code="LINE_DENSITY",
        layer="line",
        severity=92.0,
        position="整幅",
        phenomenon="临摹几乎是空白：样张上的线条基本没画出来。",
        quantity=f"临摹墨量只有样张的 {pct:.0f}%（{cp_ink} vs {ref_ink}px）",
        suggestion="先用轻线把大形和主要轮廓整体铺一遍，确认位置对了再逐区加深；不要一上来只抠局部细节。",
        recheck="并排看两幅：临摹上应能看出同样的主体轮廓，而不是大片空白。",
        confidence=0.95,
        metric={"global": True, "copy_ink": cp_ink, "ref_ink": ref_ink},
        region=(-1, -1),
    )


def layer_coverage_note(cov: np.ndarray, align_ok: bool, align_msg: str) -> List[str]:
    """生成"判不了"的说明行，供报告与 CLI 使用。"""
    notes: List[str] = []
    if not align_ok:
        notes.append(f"⚠ 无法可靠比对：配准未通过（{align_msg}）。"
                     f"以下结论只能当「大概」看，请重拍正面、四角对齐后重试。")
    low = [(r, c) for r in range(cov.shape[0]) for c in range(cov.shape[1]) if cov[r, c] < 0.5]
    if low:
        names = "、".join(cell_label(r, c) for r, c in low[:4])
        notes.append(f"⚠ {names} 有效像素不足一半（超出画布或遮挡），该区不参与判定。")
    if cov.mean() < 0.55:
        notes.append("⚠ 临摹画布与样张差异较大，建议重拍正面、四角对齐后重试。")
    return notes
