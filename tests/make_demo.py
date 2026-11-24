"""生成 samples/ 里的演示图（不需要外部素材，可重复生成）。

用法：python tests\\make_demo.py
- samples/demo_ref.png   样张（线稿人形）
- samples/demo_copy.png  临摹：少画一条腿 + 多画一条横线 + 右下画灰了
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, ROOT)

from run_checks import figure, paper, soft_rect          # noqa: E402
from sketchfix.io_prep import imwrite_unicode            # noqa: E402

SAMPLES = os.path.join(ROOT, "samples")


def main() -> int:
    os.makedirs(SAMPLES, exist_ok=True)
    ref = figure(paper(21))
    copy = figure(paper(22), erase_leg=True, extra_line=True)
    copy = soft_rect(copy, 470, 320, 700, 520, -30)
    imwrite_unicode(os.path.join(SAMPLES, "demo_ref.png"), ref)
    imwrite_unicode(os.path.join(SAMPLES, "demo_copy.png"), copy)
    print("已生成: samples\\demo_ref.png, samples\\demo_copy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
