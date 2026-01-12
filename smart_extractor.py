#!/usr/bin/env python3
"""智能截取策略 - 基于医学文献结构的高效提取"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Section:
    name: str
    start: int
    end: int
    content: str
    priority: int  # 1=必须全保留, 2=保留首尾, 3=可跳过

# 章节标题模式 (中英文)
SECTION_PATTERNS = [
    (r'(?i)^#{0,3}\s*(abstract|摘\s*要|summary)\s*[:：]?\s*$', 'abstract', 1),
    (r'(?i)^#{0,3}\s*(results?|结\s*果)\s*[:：]?\s*$', 'results', 1),
    (r'(?i)^#{0,3}\s*(conclusions?|结\s*论)\s*[:：]?\s*$', 'conclusion', 1),
    (r'(?i)^#{0,3}\s*(recommendations?|推\s*荐|建\s*议)\s*[:：]?\s*$', 'recommendations', 1),
    (r'(?i)^#{0,3}\s*(key\s*findings?|主要发现|要\s*点)\s*[:：]?\s*$', 'key_findings', 1),
    (r'(?i)^#{0,3}\s*(introduction|引\s*言|背\s*景|前\s*言)\s*[:：]?\s*$', 'introduction', 2),
    (r'(?i)^#{0,3}\s*(discussion|讨\s*论)\s*[:：]?\s*$', 'discussion', 2),
    (r'(?i)^#{0,3}\s*(limitations?|局限性?|不足)\s*[:：]?\s*$', 'limitations', 2),
    (r'(?i)^#{0,3}\s*(methods?|方\s*法|材料与方法|研究方法)\s*[:：]?\s*$', 'methods', 3),
    (r'(?i)^#{0,3}\s*(patients?|subjects?|participants?|研究对象|病例)\s*[:：]?\s*$', 'subjects', 3),
]

TABLE_PATTERN = re.compile(r'(Table|表|Tab\.?)\s*(\d+|[一二三四五六七八九十]+)[.．、:\s]*([^\n]{0,100})', re.IGNORECASE)
FIGURE_PATTERN = re.compile(r'(Figure|Fig\.?|图)\s*(\d+|[一二三四五六七八九十]+)[.．、:\s]*([^\n]{0,100})', re.IGNORECASE)

STAT_KEYWORDS = [r'p\s*[<>=]\s*0\.\d+', r'CI\s*[:：]?\s*\d', r'HR\s*[:：=]?\s*\d', r'OR\s*[:：=]?\s*\d',
    r'95%', r'significant', r'显著', r'median|mean|中位|平均', r'n\s*=\s*\d+']
CONCLUSION_KEYWORDS = [r'conclude|结论|总之', r'recommend|建议|推荐', r'suggest|提示|表明', r'effective|有效']


def detect_sections(text: str) -> List[Section]:
    lines = text.split('\n')
    sections = []
    current_section = None
    current_start = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        for pattern, name, priority in SECTION_PATTERNS:
            if re.match(pattern, line_stripped):
                if current_section:
                    sections.append(Section(current_section[0], current_start, i, 
                        '\n'.join(lines[current_start:i]), current_section[1]))
                current_section = (name, priority)
                current_start = i
                break
    
    if current_section:
        sections.append(Section(current_section[0], current_start, len(lines),
            '\n'.join(lines[current_start:]), current_section[1]))
    return sections


def extract_tables_and_figures(text: str) -> str:
    extracted = []
    for match in TABLE_PATTERN.finditer(text):
        start, end = match.end(), min(match.end() + 1500, len(text))
        next_sec = re.search(r'\n\s*(Table|表|Figure|图|\n\n\n)', text[start:end])
        if next_sec:
            end = start + next_sec.start()
        extracted.append(f"[表格] {match.group(0)}\n{text[start:end].strip()}")
    
    for match in FIGURE_PATTERN.finditer(text):
        caption = text[match.end():match.end()+200].split('\n')[0].strip()
        if caption:
            extracted.append(f"[图注] {match.group(0)} {caption}")
    return '\n\n'.join(extracted)


def calculate_importance(paragraph: str) -> float:
    score = 0.0
    for kw in STAT_KEYWORDS:
        if re.search(kw, paragraph, re.IGNORECASE):
            score += 2.0
    for kw in CONCLUSION_KEYWORDS:
        if re.search(kw, paragraph, re.IGNORECASE):
            score += 1.5
    if len(re.findall(r'\d+\.?\d*', paragraph)) > 3:
        score += 1.0
    if len(paragraph) < 100:
        score *= 0.5
    return score


def extract_important_paragraphs(text: str, max_chars: int = 5000) -> str:
    paragraphs = re.split(r'\n\s*\n', text)
    scored = [(p, calculate_importance(p)) for p in paragraphs if len(p.strip()) > 50]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    result, total = [], 0
    for para, score in scored:
        if score < 1.0 or total + len(para) > max_chars:
            break
        result.append(para)
        total += len(para)
    return '\n\n'.join(result)


def smart_extract(text: str, max_output_chars: int = 50000) -> Tuple[str, Dict]:
    """智能截取主函数"""
    original_len = len(text)
    stats = {'original_chars': original_len, 'sections_found': [], 'tables_found': 0, 'figures_found': 0}
    
    if original_len <= max_output_chars:
        stats['extracted_chars'] = original_len
        stats['compression_ratio'] = 1.0
        return text, stats
    
    parts = []
    
    # 1. 开头 8K
    parts.append(("header", text[:8000]))
    
    # 2. 按章节优先级提取
    sections = detect_sections(text)
    stats['sections_found'] = [s.name for s in sections]
    
    for s in sections:
        if s.priority == 1:
            parts.append((s.name, s.content[:15000]))
        elif s.priority == 2:
            if len(s.content) > 6000:
                parts.append((s.name, s.content[:3000] + "\n...[省略]...\n" + s.content[-3000:]))
            else:
                parts.append((s.name, s.content))
        else:
            parts.append((s.name, s.content[:1500]))
    
    # 3. 表格图注
    tf = extract_tables_and_figures(text)
    if tf:
        stats['tables_found'] = len(TABLE_PATTERN.findall(text))
        stats['figures_found'] = len(FIGURE_PATTERN.findall(text))
        parts.append(("tables_figures", tf[:10000]))
    
    # 4. 高重要性段落
    current_len = sum(len(p[1]) for p in parts)
    if current_len < max_output_chars - 5000:
        imp = extract_important_paragraphs(text, max_output_chars - current_len - 1000)
        if imp:
            parts.append(("important", imp))
    
    # 5. 结尾
    ref_match = re.search(r'(?i)(references?|参考文献)\s*\n', text)
    footer_start = ref_match.start() - 5000 if ref_match else len(text) - 5000
    parts.append(("footer", text[max(0, footer_start):ref_match.start() if ref_match else len(text)]))
    
    result = "\n\n---\n\n".join([f"[{n.upper()}]\n{c}" for n, c in parts])
    if len(result) > max_output_chars:
        result = result[:max_output_chars]
    
    stats['extracted_chars'] = len(result)
    stats['compression_ratio'] = round(len(result) / original_len, 2)
    return result, stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/root/pdf_summarization_deploy_20251225_093847/src")
    from pdf_summarizer.utils.pdf_parser import PDFParser
    
    parser = PDFParser()
    text, _ = parser.extract_text_from_file("/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460835542_8106296.pdf")
    
    print(f"原文: {len(text)} 字符")
    extracted, stats = smart_extract(text, 50000)
    print(f"截取后: {stats['extracted_chars']} 字符 (压缩比: {stats['compression_ratio']})")
    print(f"章节: {stats['sections_found']}")
    print(f"表格: {stats['tables_found']}, 图: {stats['figures_found']}")
