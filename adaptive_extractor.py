#!/usr/bin/env python
from __future__ import annotations
"""
自适应医学文献提取器 - 两阶段处理
阶段1：分类器识别文献类型
阶段2：选择专项提取器处理
"""

import argparse, json, os, re, sys, time, asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

# 提示词目录
PROMPT_DIR = Path("/root/提示词")
CLASSIFIER_PROMPT = (PROMPT_DIR / "分类器/classifier_prompt.md").read_text(encoding="utf-8")

# 类型到提取器映射
EXTRACTOR_MAP = {
    "RCT": "RCT_extractor_v2.md",
    "META": "META_extractor_v2.md",
    "REPORT": "REPORT_extractor.md",
    "GUIDELINE": "GUIDELINE_extractor_v4.md",
    "COHORT": "COHORT_extractor_v2.md",
    "CASE_CONTROL": "CASE_CONTROL_extractor.md",
    "CROSS_SECTIONAL": "CROSS_SECTIONAL_extractor.md",
    "CASE_REPORT": "CASE_REPORT_extractor.md",
    "DIAGNOSTIC": "DIAGNOSTIC_extractor.md",
    "REVIEW": "REVIEW_extractor.md",
    "OTHER": "OTHER_extractor_v2.md",
}

@dataclass
class ExtractionResult:
    pdf_name: str
    pdf_pages: int
    pdf_chars: int
    doc_type: str = ""
    doc_type_confidence: str = ""
    classification_time: float = 0.0
    extraction_time: float = 0.0
    total_tokens: int = 0
    success: bool = False
    error: str = ""
    classification_result: Dict = field(default_factory=dict)
    extraction_result: Dict = field(default_factory=dict)

def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    parser = PDFParser(max_pages=200)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
    return full_text, metadata

def extract_json_object(text: str) -> Dict[str, Any]:
    # 移除 <think>...</think> 块
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # 移除 markdown 代码块
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m: text = m.group(1)
    
    start = text.find("{")
    if start < 0: raise ValueError("No JSON found")
    
    in_str, esc, depth, end = False, False, 0, None
    for i in range(start, len(text)):
        ch = text[i]
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i + 1; break
    if end is None: raise ValueError("Unclosed JSON")
    raw = text[start:end]
    try: return json.loads(raw)
    except: return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))

def call_qwen_vllm(prompt: str, text: str, enable_thinking: bool = False, 
                   base_url: str = "http://localhost:8000/v1", max_tokens: int = 8192) -> Tuple[str, int]:
    """调用 vLLM Qwen"""
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": prompt + text}],
        temperature=0.1,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    )
    
    output = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return output, tokens

def _build_classification_text(full_text: str, max_chars: int = 12000) -> Tuple[str, list[int]]:
    """方案A：自适应选页用于分类。

    目标：避免只喂封面/版权页，优先选取信息密度更高、包含研究/指标信号的页面。
    """

    matches = list(re.finditer(r"\[p(\d+)\]", full_text))
    if not matches:
        return full_text[:max_chars], []

    pages: Dict[int, str] = {}
    for i, m in enumerate(matches):
        pno = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        if content:
            pages[pno] = content

    if not pages:
        return full_text[:max_chars], []

    # 关键词/线索
    report_cues = re.compile(
        r"\bISBN\b|Suggested citation|World Health Organization|\bWHO\b|\bAtlas\b|Annual report|Global report|Surveillance|Monitoring|\bcitation\b",
        re.I,
    )
    prevalence_cues = re.compile(r"prevalence|患病率|流行率", re.I)
    survey_cues = re.compile(r"\bsurvey\b|questionnaire|问卷|调查", re.I)

    percent_pat = re.compile(r"\b\d{1,3}(?:\.\d+)?%\b")
    per100k_pat = re.compile(r"per\s*100[, ]?000", re.I)

    boilerplate = re.compile(r"Creative Commons|All rights reserved|licen[cs]e|disclaimer|mediation", re.I)

    # 评分：用于补充选页（不是最终分类规则）
    kw_patterns = [
        # 研究设计
        (re.compile(r"\brandomi[sz]ed\b|\btrial\b|\bdouble[- ]blind\b", re.I), 6),
        (re.compile(r"\bPRISMA\b|forest plot|\bI\s*\^?2\b|heterogene", re.I), 6),
        (re.compile(r"\bcohort\b|hazard ratio|\bHR\b|relative risk|\bRR\b", re.I), 5),
        (re.compile(r"case[- ]control|odds ratio|\bOR\b", re.I), 5),
        (re.compile(r"cross[- ]sectional|\bsurvey\b|questionnaire", re.I), 4),
        (re.compile(r"prevalence|incidence", re.I), 6),
        # 中文关键词
        (re.compile(r"随机|盲法|对照试验"), 5),
        (re.compile(r"系统评价|Meta分析|森林图|异质性|I\s*\^?2"), 5),
        (re.compile(r"队列|随访|HR|RR"), 4),
        (re.compile(r"病例对照|OR"), 4),
        (re.compile(r"横断面|调查|患病率|流行率"), 5),
        # 报告/Atlas
        (report_cues, 3),
    ]

    def score_page(t: str) -> float:
        t = t.strip()
        if not t:
            return -1.0

        digits = len(re.findall(r"\d", t))
        perc = len(percent_pat.findall(t))
        per100k = len(per100k_pat.findall(t))

        kw = 0
        for pat, w in kw_patterns:
            kw += w * len(pat.findall(t))

        boil = len(boilerplate.findall(t))

        # 数字/指标/方法学页优先；同时避免纯版权页占满
        return (digits * 0.08) + (perc * 4.0) + (per100k * 8.0) + kw - (boil * 2.0) + min(len(t), 4000) / 2500

    def _count(pat: re.Pattern, t: str) -> int:
        return len(pat.findall(t))

    def _pick_best(cands: list[int], key_fn):
        best_p = None
        best_score = 0
        for p in cands:
            v = key_fn(p)
            if v > best_score:
                best_score = v
                best_p = p
        return best_p, best_score

    first_page = min(pages.keys())

    # 只在前 N 页里做候选池（兼顾性能与“目录后开始有数据”的报告）
    POOL_MAX_PAGE = 40
    pool = [p for p in pages.keys() if p != first_page and p <= POOL_MAX_PAGE]
    if len(pool) < 6:
        pool = [p for p in pages.keys() if p != first_page and p <= 80]

    must = {first_page}

    # (1) 身份页：在前10页里找 report 线索（ISBN/citation/Atlas/WHO 等）
    id_pool = [p for p in pages.keys() if p != first_page and p <= 10]
    rp, rscore = _pick_best(id_pool, lambda p: _count(report_cues, pages[p]))
    if rp and rscore > 0:
        must.add(rp)

    # (2) prevalence 页（如存在）
    pp, pscore = _pick_best(pool, lambda p: _count(prevalence_cues, pages[p]))
    if pp and pscore > 0:
        must.add(pp)

    # (3) 指标页：优先 per 100,000 / % / 数字密度
    np, nscore = _pick_best(
        pool,
        lambda p: (_count(per100k_pat, pages[p]) * 10)
        + (_count(percent_pat, pages[p]) * 5)
        + (len(re.findall(r"\d", pages[p])) * 0.02),
    )
    if np and nscore > 0:
        must.add(np)

    # (4) survey/问卷 页（如存在）
    sp, sscore = _pick_best(pool, lambda p: _count(survey_cues, pages[p]))
    if sp and sscore > 0:
        must.add(sp)

    # (5) 其余页按 score 补齐
    scored = sorted(((score_page(pages[p]), p) for p in pool), reverse=True)
    selected = list(sorted(must))

    MAX_PAGES = 6
    for _, p in scored:
        if p in must:
            continue
        selected.append(p)
        if len(selected) >= MAX_PAGES:
            break

    selected = sorted(set(selected))

    # 组装文本（按页码顺序）
    out = []
    used_pages: list[int] = []
    total = 0
    for p in selected:
        chunk = f"[p{p}]\n{pages.get(p, '').strip()}\n\n"
        if not chunk.strip():
            continue

        if total + len(chunk) > max_chars:
            remain = max_chars - total
            if remain > 200:
                out.append(chunk[:remain])
                used_pages.append(p)
            break

        out.append(chunk)
        used_pages.append(p)
        total += len(chunk)

    final_text = "".join(out).strip()
    if not final_text:
        return full_text[:max_chars], []

    return final_text, used_pages


def classify_document(text: str, enable_thinking: bool = False) -> Tuple[Dict, int]:
    """阶段1：分类（方案A：自适应选页）"""
    try:
        cls_text, used_pages = _build_classification_text(text, max_chars=12000)
    except Exception:
        cls_text, used_pages = (text[:12000], [])

    if used_pages:
        print(f"    分类选页: {used_pages} (chars={len(cls_text)})")

    output, tokens = call_qwen_vllm(CLASSIFIER_PROMPT, cls_text, enable_thinking, max_tokens=1024)
    result = extract_json_object(output)
    return result, tokens


def extract_with_type(
    text: str,
    doc_type: str,
    enable_thinking: bool = False,
    classification: Dict[str, Any] | None = None,
) -> Tuple[Dict, int]:
    """阶段2：专项提取"""
    extractor_file = EXTRACTOR_MAP.get(doc_type, "OTHER_extractor.md")
    extractor_path = PROMPT_DIR / "专项提取器" / extractor_file

    if not extractor_path.exists():
        extractor_path = PROMPT_DIR / "专项提取器/OTHER_extractor.md"

    extractor_prompt = extractor_path.read_text(encoding="utf-8")

    # 注入上游分类器信息（v4.3 输出: type/conf/evidence/meta/route/loc）
    upstream_block = ""
    if classification:
        slim = {}
        for k in ["type", "conf", "evidence", "meta", "route", "loc", "doc_type", "confidence"]:
            if k in classification and classification.get(k) is not None:
                slim[k] = classification.get(k)
        if slim:
            upstream_block = (
                "【上游分类器输出（仅供参考；如与正文冲突，以正文为准）】\n"
                + json.dumps(slim, ensure_ascii=False, indent=2)
                + "\n\n"
            )

    base_prompt = upstream_block + extractor_prompt

    # 长文档分块处理 - 优化版：增大块大小 + 并行处理
    MAX_CHUNK = 28000  # 字符数
    OVERLAP = 3000  # 重叠区
    MAX_PARALLEL = 3  # 最大并行数

    def _is_empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip() in {"", "Not Reported", "Not Available", "N/A", "Not Applicable"}:
            return True
        if isinstance(v, (list, dict)) and len(v) == 0:
            return True
        return False

    def _merge_lists(a: list, b: list) -> list:
        out = []
        seen = set()
        for item in a + b:
            if isinstance(item, dict):
                key = ("dict", json.dumps(item, sort_keys=True, ensure_ascii=False))
            else:
                try:
                    key = ("scalar", item)
                except TypeError:
                    key = ("repr", repr(item))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _deep_merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            merged = dict(a)
            for k, bv in b.items():
                if k in merged:
                    merged[k] = _deep_merge(merged[k], bv)
                else:
                    merged[k] = bv
            return merged
        if isinstance(a, list) and isinstance(b, list):
            return _merge_lists(a, b)

        # scalar / mismatched types: keep first non-empty
        if _is_empty(a) and not _is_empty(b):
            return b
        return a

    if len(text) <= MAX_CHUNK:
        output, tokens = call_qwen_vllm(base_prompt, text, enable_thinking, max_tokens=8192)
        return extract_json_object(output), tokens

    # 分块处理
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK, len(text))
        if end < len(text):
            # 在句号/段落处断开
            for sep in ["\n\n", "。\n", ".\n", "\n"]:
                idx = text.rfind(sep, start + MAX_CHUNK // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - OVERLAP if end < len(text) else end

    print(f"    分块处理: {len(chunks)} 块 (并行度: {min(len(chunks), MAX_PARALLEL)})")

    def process_chunk(args):
        i, chunk = args
        chunk_prompt = f"这是文档的第 {i+1}/{len(chunks)} 部分（类型: {doc_type}）。\n\n{base_prompt}"
        try:
            output, tokens = call_qwen_vllm(chunk_prompt, chunk, enable_thinking, max_tokens=7000)
            result = extract_json_object(output)
            return (i, result, tokens, None)
        except Exception as e:
            return (i, None, 0, str(e))

    all_results = [None] * len(chunks)
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = list(executor.map(process_chunk, enumerate(chunks)))

    for i, result, tokens, error in futures:
        if error:
            print(f"    块 {i+1} 失败: {error}")
        else:
            all_results[i] = result
            total_tokens += tokens

    all_results = [r for r in all_results if r is not None]
    if not all_results:
        return {}, total_tokens

    merged = all_results[0]
    for r in all_results[1:]:
        merged = _deep_merge(merged, r)

    return merged, total_tokens

def process_pdf(pdf_path: Path, output_dir: Path, enable_thinking: bool = False) -> ExtractionResult:
    """处理单个 PDF"""
    result = ExtractionResult(pdf_name=pdf_path.name, pdf_pages=0, pdf_chars=0)
    
    # 解析 PDF
    try:
        full_text, metadata = extract_pdf_text(str(pdf_path))
        result.pdf_pages = metadata.get("processed_pages", 0)
        result.pdf_chars = len(full_text)
    except Exception as e:
        result.error = f"PDF解析失败: {e}"
        return result
    
    if result.pdf_chars < 500:
        result.error = "PDF内容过短"
        return result
    
    # 阶段1：分类
    print(f"  [阶段1] 分类中...")
    t0 = time.time()
    try:
        classification, cls_tokens = classify_document(full_text, enable_thinking)
        result.classification_time = time.time() - t0
        result.classification_result = classification
        result.doc_type = classification.get("type") or classification.get("doc_type") or "OTHER"
        result.doc_type_confidence = classification.get("conf") or classification.get("confidence") or "low"
        result.total_tokens = cls_tokens
        print(f"    类型: {result.doc_type} (置信度: {result.doc_type_confidence})")
    except Exception as e:
        result.error = f"分类失败: {e}"
        result.doc_type = "OTHER"
    
    # 阶段2：专项提取
    print(f"  [阶段2] 使用 {result.doc_type} 提取器...")
    t0 = time.time()
    try:
        extraction, ext_tokens = extract_with_type(full_text, result.doc_type, enable_thinking, classification=classification)
        result.extraction_time = time.time() - t0
        result.extraction_result = extraction
        result.total_tokens += ext_tokens
        result.success = True
        print(f"    提取完成: {result.extraction_time:.1f}s")
    except Exception as e:
        result.error = f"提取失败: {e}"
    
    # 保存结果
    mode_suffix = "thinking" if enable_thinking else "no_thinking"
    
    # 分类结果
    cls_path = output_dir / f"{pdf_path.stem}.classification.json"
    cls_path.write_text(json.dumps(result.classification_result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # 提取结果
    ext_path = output_dir / f"{pdf_path.stem}.{result.doc_type}.{mode_suffix}.json"
    ext_path.write_text(json.dumps(result.extraction_result, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return result

def main():
    ap = argparse.ArgumentParser(description="自适应医学文献提取器")
    ap.add_argument("pdf", nargs="?", help="单个PDF文件路径")
    ap.add_argument("--pdf-dir", help="PDF目录")
    ap.add_argument("--output-dir", default="/root/adaptive_extraction_results")
    ap.add_argument("--thinking", action="store_true", help="启用思考模式")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集PDF
    pdfs = []
    if args.pdf:
        pdfs = [Path(args.pdf)]
    elif args.pdf_dir:
        pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
        if args.limit > 0:
            pdfs = pdfs[:args.limit]
    else:
        print("请指定 PDF 文件或目录")
        return
    
    print(f"PDF数量: {len(pdfs)}")
    print(f"模式: {'思考' if args.thinking else '非思考'}")
    print("-" * 50)
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        result = process_pdf(pdf, output_dir, args.thinking)
        results.append(result)
        
        if result.success:
            print(f"  ✓ {result.doc_type} | 分类{result.classification_time:.1f}s + 提取{result.extraction_time:.1f}s | {result.total_tokens} tokens")
        else:
            print(f"  ✗ {result.error}")
    
    # 保存汇总
    summary = {
        "total": len(results),
        "success": sum(r.success for r in results),
        "by_type": {},
        "results": [{k: v for k, v in vars(r).items() if k not in ["classification_result", "extraction_result"]} for r in results]
    }
    for r in results:
        if r.doc_type:
            summary["by_type"][r.doc_type] = summary["by_type"].get(r.doc_type, 0) + 1
    
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 50)
    print(f"完成: {summary['success']}/{summary['total']}")
    print(f"类型分布: {summary['by_type']}")
    print(f"结果目录: {output_dir}")

if __name__ == "__main__":
    main()
