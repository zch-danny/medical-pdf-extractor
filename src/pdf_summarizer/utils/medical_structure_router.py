"""医学文献结构路由

将一篇医学文献的全文文本（带页码标记，如 [p12]）尽量按论文结构切分为：
- Design: Abstract/Introduction/Methods（设计与方法学）
- Evidence: Results（结果与证据，含表格/图注）
- Conclusion: Discussion/Conclusion/Limitations（讨论与结论）

注意：这是启发式切分，不保证所有论文都严格匹配；若未找到明确标题，将回退到按比例切分。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RoutedSections:
    design: str
    evidence: str
    conclusion: str
    debug: Dict[str, object]


# 常见章节标题（英文/中文），用于启发式定位
_RESULTS_PATTERNS = [
    r"^results\b",
    r"^findings\b",
    r"^outcomes\b",
    r"^primary\s+outcome\b",
    r"^secondary\s+outcome\b",
    r"^结果\b",
    r"^研究结果\b",
]

_DISCUSSION_PATTERNS = [
    r"^discussion\b",
    r"^conclusion\b",
    r"^conclusions\b",
    r"^limitations\b",
    r"^summary\b",
    r"^讨论\b",
    r"^结论\b",
    r"^局限\b",
    r"^局限性\b",
]


def _is_page_marker(line: str) -> bool:
    return bool(re.match(r"^\s*\[(?:p\d+|Page\s+\d+)\]\s*$", line))


def _normalize_heading(line: str) -> str:
    """Normalize heading-like lines to improve matching."""
    s = line.strip()
    # 去掉前置编号/符号
    s = re.sub(r"^\s*(?:#+\s*)?", "", s)
    s = re.sub(r"^\s*(?:\d+(?:\.\d+)*\s*[\)\.]?\s*)", "", s)
    s = s.strip()
    # 去掉结尾冒号
    s = re.sub(r"[:：]\s*$", "", s).strip()
    return s


def _find_first_heading_index(lines: List[str], patterns: List[str]) -> Optional[int]:
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if _is_page_marker(line):
            continue
        # 只把“短行”当作标题候选，避免正文误判
        if len(line) > 120:
            continue
        h = _normalize_heading(line)
        for pat in patterns:
            if re.match(pat, h, flags=re.IGNORECASE):
                return i
    return None


def route_medical_sections(text: str) -> RoutedSections:
    """将全文按结构切分为 design / evidence / conclusion 三段。

    Args:
        text: 全文文本（建议包含 [pX] 页码标记）

    Returns:
        RoutedSections
    """
    lines = text.splitlines()

    idx_results = _find_first_heading_index(lines, _RESULTS_PATTERNS)
    idx_discussion = _find_first_heading_index(lines, _DISCUSSION_PATTERNS)

    n = len(lines)

    # 回退：按比例切分
    fallback_results = max(0, int(n * 0.40))
    fallback_discussion = max(0, int(n * 0.75))

    if idx_results is None:
        idx_results = fallback_results
    if idx_discussion is None:
        idx_discussion = fallback_discussion

    # 修正顺序
    if idx_discussion < idx_results:
        idx_discussion = min(n, idx_results + max(0, int(n * 0.20)))

    design = "\n".join(lines[:idx_results]).strip()
    evidence = "\n".join(lines[idx_results:idx_discussion]).strip()
    conclusion = "\n".join(lines[idx_discussion:]).strip()

    debug = {
        "total_lines": n,
        "idx_results": idx_results,
        "idx_discussion": idx_discussion,
        "fallback_results": fallback_results,
        "fallback_discussion": fallback_discussion,
    }

    return RoutedSections(design=design, evidence=evidence, conclusion=conclusion, debug=debug)
