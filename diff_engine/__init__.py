# -*- coding: utf-8 -*-
"""
diff_engine: Module d'analyse différentielle et de comparaison visuelle (Visual Diff).
"""
from visual_diff import compare_dxf_files, DiffResult, DiffItem
from diff_report import format_diff_report, print_diff_report
from render_diff_png import render_diff_to_png

__all__ = [
    "compare_dxf_files",
    "DiffResult",
    "DiffItem",
    "format_diff_report",
    "print_diff_report",
    "render_diff_to_png",
]
