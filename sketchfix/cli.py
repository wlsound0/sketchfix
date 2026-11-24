"""命令行入口：python -m sketchfix --ref 样张 --copy 临摹 --out out"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import advise, regions
from .align import align_manual, align_pair
from .io_prep import load_prep
from .layers import line as line_layer
from .layers import proportion as prop_layer
from .layers import value as value_layer
from .types import Finding, ensure_dir


def parse_quad(text: str) -> List[Tuple[float, float]]:
    pts = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            pts.append(float(chunk))
    if len(pts) != 8:
        raise argparse.ArgumentTypeError("四角需 8 个数：x1,y1,x2,y2,x3,y3,x4,y4（左上、右上、右下、左下）")
    return [(pts[i], pts[i + 1]) for i in range(0, 8, 2)]


def run(ref_path: str, copy_path: str, out_dir: str,
        quad: Optional[Sequence[Tuple[float, float]]] = None,
        strictness: float = 1.0, top_n: int = 5,
        model: str = "similarity") -> dict:
    ref = load_prep(ref_path)
    copy = load_prep(copy_path)

    ar = align_manual(ref, copy, quad) if quad else align_pair(ref, copy, model=model)
    valid = ar.valid
    notes = regions.layer_coverage_note(
        regions.grid_coverage(valid), ar.ok, ar.message or "")
    if getattr(ar, "warn", ""):
        notes.insert(0, ar.warn)

    findings: List[Finding] = []
    blank = regions.blank_copy_finding(ref.fg, ar.warped_fg)
    if blank is not None:
        findings.append(blank)
    findings += prop_layer.compare_proportion(ref.fg, ar.warped_fg, valid)
    findings += line_layer.compare_lines(ref.gray, ar.warped_gray, valid)
    findings += value_layer.compare_value(ref.gray, ar.warped_gray, valid)

    if blank is not None:
        # 整幅"几乎空白"已经说明问题，逐区"墨量偏少"是同一件事的碎话。
        findings = [f for f in findings
                    if f.metric.get("global") or f.code != "LINE_DENSITY"]

    if not ar.ok:
        # 配准都不可靠时，逐区结论（某区缺线/某区偏暗）没有意义，
        # 只保留整体性结论，避免用假精确的数字误导用户。
        findings = [f for f in findings
                    if f.layer == "proportion" or f.metric.get("global")]
        if blank is not None:
            # 「临摹几乎空白」本身就是可靠结论：仍要说"无法可靠比对"
            # （逐区结论确实无从谈起），但把"重拍对齐"那句换成真正该做的事。
            notes = [f"⚠ 无法可靠比对：配准未通过（{ar.message}）。"
                     "临摹几乎是空白，逐区比对（缺哪段线、哪块偏暗）无从谈起，"
                     "只给出这一条整体结论：先把大形和主要轮廓铺出来。"]

    findings = regions.merge_adjacent(findings)
    findings, derived_notes = regions.suppress_derived(findings)
    notes = notes + derived_notes
    findings = regions.rank(findings, top_n=top_n, strictness=strictness)

    align_info = {
        "ok": bool(ar.ok),
        "inlier_ratio": float(ar.inlier_ratio),
        "residual_px": float(ar.residual_px),
        "message": ar.message,
        "model": getattr(ar, "model", "homography"),
        "warn": getattr(ar, "warn", ""),
    }
    paths = advise.build_report(ref_path, copy_path, align_info, findings, notes,
                                out_dir, ref_bgr=ref.bgr)
    ar_path = os.path.join(out_dir, "aligned_copy.png")
    cv2.imencode(".png", ar.warped)[1].tofile(ar_path)
    paths["aligned_copy"] = ar_path
    return {"paths": paths, "findings": findings, "align": align_info, "notes": notes}


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Windows 控制台默认 GBK，打印 "⚠" 之类字符会直接抛 UnicodeEncodeError，
    # 这里改成 utf-8 + 替换，坏字符不至于把整个诊断打断。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(prog="sketchfix", description="临摹对比诊断（纯本地几何版）")
    ap.add_argument("--ref", required=True, help="样张图片路径")
    ap.add_argument("--copy", required=True, help="临摹图片路径")
    ap.add_argument("--out", default="out", help="输出目录")
    ap.add_argument("--manual-quad", type=parse_quad, default=None,
                    help="手动指定临摹画布四角: 左上,右上,右下,左下 共8个数")
    ap.add_argument("--strictness", type=float, default=1.0, help=">1 更严格，<1 更宽松")
    ap.add_argument("--top", type=int, default=5, help="最多输出几条")
    ap.add_argument("--align", choices=["similarity", "homography"],
                    default="similarity",
                    help="similarity=只用旋转+等比缩放+平移（默认，能查出画得比例不对）；"
                         "homography=允许透视，仅当纸张拍歪了才用")
    args = ap.parse_args(argv)

    for p in (args.ref, args.copy):
        if not os.path.exists(p):
            print(f"[错误] 找不到文件: {p}", file=sys.stderr)
            return 2
    ensure_dir(args.out)

    try:
        res = run(args.ref, args.copy, args.out, args.manual_quad,
                  args.strictness, args.top, args.align)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # 不允许半途崩掉用户的任务
        print(f"[错误] 处理失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"配准: {'通过' if res['align']['ok'] else '未通过'} "
          f"(模型 {res['align']['model']}, 内点率 {res['align']['inlier_ratio']:.2f}, "
          f"残差 {res['align']['residual_px']:.2f}px)")
    if res["notes"]:
        for n in res["notes"]:
            print(n)
    print(f"问题: {len(res['findings'])} 条")
    for i, f in enumerate(res["findings"], start=1):
        print(f"  {i}. [{f.severity:.0f}] {f.layer_name}·{f.position} — {f.phenomenon}（{f.quantity}）")
    print(f"报告: {res['paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
