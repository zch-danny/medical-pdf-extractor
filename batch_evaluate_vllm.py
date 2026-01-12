#!/usr/bin/env python
"""
批量评测脚本 - 支持 vLLM API + 思考/非思考模式对比
使用方案1（动态分块）+ 方案3（vLLM部署）
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

# ========== 方案1: 动态分块配置 ==========
MAX_CHUNK_CHARS = 12000   # 每块最大字符数（约 4K tokens）
OVERLAP_CHARS = 2000      # 块间重叠字符数

# ========== 提示词 ==========
PROMPT_PATH = Path("/root/提示词/v2提示词.md")
EXTRACTION_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

@dataclass
class EvalResult:
    pdf_name: str
    pdf_pages: int
    pdf_chars: int
    gemini_time: float = 0.0
    qwen_thinking_time: float = 0.0
    qwen_no_thinking_time: float = 0.0
    gemini_success: bool = False
    qwen_thinking_success: bool = False
    qwen_no_thinking_success: bool = False
    gemini_error: str = ""
    qwen_thinking_error: str = ""
    qwen_no_thinking_error: str = ""
    gemini_tokens: Dict[str, int] = field(default_factory=dict)
    qwen_thinking_tokens: Dict[str, int] = field(default_factory=dict)
    qwen_no_thinking_tokens: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

def extract_pdf_text(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    parser = PDFParser(max_pages=200)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
    return full_text, metadata

def get_page_markers(text: str) -> List[str]:
    return re.findall(r"\[p(\d+)\]", text)

def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """方案1: 动态分块 + 滑动窗口"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # 尝试在句号/换行处断开
        if end < len(text):
            for sep in ['\n\n', '。\n', '.\n', '\n', '。', '.']:
                idx = text.rfind(sep, start + max_chars // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks

def call_gemini_api(text: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """调用 Gemini API (OpenAI 兼容)"""
    from openai import OpenAI
    api_key = os.environ.get("GEMINI_API_KEY", "")
    api_domain = os.environ.get("GEMINI_API_DOMAIN", "api.bltcy.ai")
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    
    client = OpenAI(api_key=api_key, base_url=f"https://{api_domain}/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}],
        temperature=0.1,
        max_tokens=8192,
    )
    tokens = {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens, "total": response.usage.total_tokens}
    return extract_json_object(response.choices[0].message.content), tokens

def call_qwen_vllm(text: str, enable_thinking: bool, base_url: str = "http://localhost:8000/v1") -> Tuple[Dict[str, Any], Dict[str, int]]:
    """调用 vLLM 部署的 Qwen3-8B"""
    from openai import OpenAI
    
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    # 如果文本过长，使用分块处理
    chunks = chunk_text(text)
    
    if len(chunks) == 1:
        # 单块直接处理
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}],
            temperature=0.1,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        )
        tokens = {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens, "total": response.usage.total_tokens}
        return extract_json_object(response.choices[0].message.content), tokens
    
    # 多块 Map-Reduce 处理
    print(f"    分块处理: {len(chunks)} 块")
    all_results = []
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"这是文档的第 {i+1}/{len(chunks)} 部分。\n\n{EXTRACTION_PROMPT}{chunk}"
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "user", "content": chunk_prompt}],
            temperature=0.1,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        )
        total_tokens["prompt"] += response.usage.prompt_tokens
        total_tokens["completion"] += response.usage.completion_tokens
        total_tokens["total"] += response.usage.total_tokens
        
        try:
            chunk_result = extract_json_object(response.choices[0].message.content)
            all_results.append(chunk_result)
        except Exception as e:
            print(f"    块 {i+1} 解析失败: {e}")
    
    # 合并结果
    merged = merge_chunk_results(all_results)
    return merged, total_tokens

def merge_chunk_results(results: List[Dict]) -> Dict:
    """合并多块结果（面向 v2 schema）"""
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    merged = results[0].copy()

    def _kr_key(kr: Dict) -> str:
        sid = str(kr.get("study_id", "") or "").strip()
        out = str(kr.get("outcome_name", "") or "").strip()
        tp = str(kr.get("timepoint", "") or "").strip()
        key = "|".join([sid, out, tp]).strip("|").lower()
        return key

    def _is_empty_str(v: object) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            s = v.strip()
            return s == "" or s == "Not Reported"
        return False

    # Helper: deep fill missing scalar fields (string only) in dicts
    def _fill_missing(dst: Dict, src: Dict, keys: List[str]):
        for k in keys:
            if k in src and (k not in dst or _is_empty_str(dst.get(k))):
                dst[k] = src.get(k)

    for r in results[1:]:
        if not isinstance(r, dict):
            continue

        # 合并 key_results（用 study_id+outcome_name+timepoint 去重）
        if isinstance(r.get("key_results"), list) and r["key_results"]:
            existing = set()
            for kr in merged.get("key_results", []) or []:
                if isinstance(kr, dict):
                    k = _kr_key(kr)
                    if k:
                        existing.add(k)
            for kr in r["key_results"]:
                if not isinstance(kr, dict):
                    continue
                k = _kr_key(kr)
                if k and k not in existing:
                    merged.setdefault("key_results", []).append(kr)
                    existing.add(k)

        # 合并 critical_appraisal
        if isinstance(r.get("critical_appraisal"), dict):
            mca = merged.setdefault("critical_appraisal", {})
            rca = r["critical_appraisal"]

            # limitations / clinical_implications / conclusions: [{item, sources}] 去重合并
            for lf in ["limitations", "clinical_implications", "conclusions"]:
                if isinstance(rca.get(lf), list) and rca[lf]:
                    m_list = mca.setdefault(lf, [])
                    existing_items = set()
                    for it in m_list:
                        if isinstance(it, dict):
                            s = str(it.get("item", "") or "").strip()
                            if s:
                                existing_items.add(s)
                    for it in rca[lf]:
                        if not isinstance(it, dict):
                            continue
                        s = str(it.get("item", "") or "").strip()
                        if s and s not in existing_items:
                            m_list.append(it)
                            existing_items.add(s)

            # bias_risk: {summary, sources} —— 以非空/非 Not Reported 的 summary 覆盖
            if isinstance(rca.get("bias_risk"), dict):
                mbr = mca.setdefault("bias_risk", {"summary": "Not Reported", "sources": []})
                rbr = rca["bias_risk"]
                if _is_empty_str(mbr.get("summary")) and not _is_empty_str(rbr.get("summary")):
                    mbr["summary"] = rbr.get("summary")
                # merge sources
                if isinstance(mbr.get("sources"), list) and isinstance(rbr.get("sources"), list):
                    mset = list(dict.fromkeys([*mbr["sources"], *rbr["sources"]]))
                    mbr["sources"] = mset

        # 尝试补全一些全局字段（不强制）
        if isinstance(r.get("doc_metadata"), dict) and isinstance(merged.get("doc_metadata"), dict):
            _fill_missing(merged["doc_metadata"], r["doc_metadata"], ["title", "authors", "source_info", "identifier"]) 
            if isinstance(r["doc_metadata"].get("funding_and_coi"), dict):
                merged.setdefault("doc_metadata", {}).setdefault("funding_and_coi", {})
                _fill_missing(merged["doc_metadata"]["funding_and_coi"], r["doc_metadata"]["funding_and_coi"], ["funding", "conflicts"]) 

        if isinstance(r.get("study_design"), dict) and isinstance(merged.get("study_design"), dict):
            _fill_missing(merged["study_design"], r["study_design"], ["primary_type", "phase", "blinding_and_control", "statistical_power"]) 

    return merged

def extract_json_object(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m: text = m.group(1)
    # 移除 <think>...</think> 块
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    start = text.find("{")
    if start < 0: raise ValueError(f"No JSON found")
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
    if end is None: raise ValueError(f"Unclosed JSON")
    raw = text[start:end]
    try: return json.loads(raw)
    except: return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))

def flatten_sources(obj, prefix=""):
    results = []
    if isinstance(obj, dict):
        if "sources" in obj and isinstance(obj["sources"], list):
            results.append((prefix, obj["sources"]))
        for k, v in obj.items():
            if k != "sources": results.extend(flatten_sources(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj): results.extend(flatten_sources(v, f"{prefix}[{i}]"))
    return results

def normalize_source(s):
    m = re.search(r"p?(\d+)", str(s), re.I)
    return f"p{m.group(1)}" if m else None

def compute_source_hit_rate(data, valid_pages):
    all_sources = flatten_sources(data)
    total, hits = 0, 0
    valid_set = set(f"p{p}" for p in valid_pages)
    for _, sources in all_sources:
        for s in sources:
            ns = normalize_source(s)
            if ns: total += 1; hits += 1 if ns in valid_set else 0
    return hits, total, hits/total if total else 1.0

def count_key_results(data):
    kr = data.get("key_results", [])
    return len(kr) if isinstance(kr, list) else 0

def evaluate_all(gemini, qwen_think, qwen_no_think, valid_pages):
    """对比三个结果"""
    metrics = {}
    
    for name, data in [("gemini", gemini), ("qwen_thinking", qwen_think), ("qwen_no_thinking", qwen_no_think)]:
        if data:
            h, t, r = compute_source_hit_rate(data, valid_pages)
            metrics[f"{name}_source_hits"] = h
            metrics[f"{name}_source_total"] = t
            metrics[f"{name}_source_rate"] = round(r, 3)
            metrics[f"{name}_key_results_count"] = count_key_results(data)
    
    return metrics

def process_single_pdf(pdf_path: Path, output_dir: Path, qwen_url: str, skip_gemini: bool, skip_qwen: bool) -> EvalResult:
    result = EvalResult(pdf_name=pdf_path.name, pdf_pages=0, pdf_chars=0)
    
    try:
        full_text, metadata = extract_pdf_text(str(pdf_path))
        result.pdf_pages = metadata.get("processed_pages", 0)
        result.pdf_chars = len(full_text)
        valid_pages = get_page_markers(full_text)
    except Exception as e:
        result.gemini_error = result.qwen_thinking_error = result.qwen_no_thinking_error = f"PDF parse failed: {e}"
        return result
    
    if result.pdf_chars < 500:
        result.gemini_error = result.qwen_thinking_error = result.qwen_no_thinking_error = "PDF too short"
        return result
    
    gemini_path = output_dir / f"{pdf_path.stem}.gemini.json"
    qwen_think_path = output_dir / f"{pdf_path.stem}.qwen_thinking.json"
    qwen_no_think_path = output_dir / f"{pdf_path.stem}.qwen_no_thinking.json"
    
    gemini_data, qwen_think_data, qwen_no_think_data = None, None, None
    
    # Gemini
    if not skip_gemini:
        try:
            t0 = time.time()
            gemini_data, tokens = call_gemini_api(full_text)
            result.gemini_time = time.time() - t0
            result.gemini_success = True
            result.gemini_tokens = tokens
            gemini_path.write_text(json.dumps(gemini_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Gemini: {result.gemini_time:.1f}s | {tokens['total']} tokens ✓")
        except Exception as e:
            result.gemini_error = str(e)[:200]
            print(f"  Gemini: FAILED - {str(e)[:80]}")
    elif gemini_path.exists():
        gemini_data = json.loads(gemini_path.read_text())
        result.gemini_success = True
        print(f"  Gemini: (cached) ✓")
    
    # Qwen Thinking
    if not skip_qwen:
        try:
            t0 = time.time()
            qwen_think_data, tokens = call_qwen_vllm(full_text, enable_thinking=True, base_url=qwen_url)
            result.qwen_thinking_time = time.time() - t0
            result.qwen_thinking_success = True
            result.qwen_thinking_tokens = tokens
            qwen_think_path.write_text(json.dumps(qwen_think_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Qwen (思考): {result.qwen_thinking_time:.1f}s | {tokens['total']} tokens ✓")
        except Exception as e:
            result.qwen_thinking_error = str(e)[:200]
            print(f"  Qwen (思考): FAILED - {str(e)[:80]}")
        
        # Qwen No-Thinking
        try:
            t0 = time.time()
            qwen_no_think_data, tokens = call_qwen_vllm(full_text, enable_thinking=False, base_url=qwen_url)
            result.qwen_no_thinking_time = time.time() - t0
            result.qwen_no_thinking_success = True
            result.qwen_no_thinking_tokens = tokens
            qwen_no_think_path.write_text(json.dumps(qwen_no_think_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Qwen (非思考): {result.qwen_no_thinking_time:.1f}s | {tokens['total']} tokens ✓")
        except Exception as e:
            result.qwen_no_thinking_error = str(e)[:200]
            print(f"  Qwen (非思考): FAILED - {str(e)[:80]}")
    else:
        if qwen_think_path.exists():
            qwen_think_data = json.loads(qwen_think_path.read_text())
            result.qwen_thinking_success = True
            print(f"  Qwen (思考): (cached) ✓")
        if qwen_no_think_path.exists():
            qwen_no_think_data = json.loads(qwen_no_think_path.read_text())
            result.qwen_no_thinking_success = True
            print(f"  Qwen (非思考): (cached) ✓")
    
    # 评估
    if any([gemini_data, qwen_think_data, qwen_no_think_data]):
        result.metrics = evaluate_all(gemini_data, qwen_think_data, qwen_no_think_data, valid_pages)
    
    return result

def generate_report(results: List[EvalResult], output_path: Path):
    lines = [
        "# 医学文献 PDF 摘要评测报告 (vLLM + 思考模式对比)",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 1. 总览",
        f"\n处理 PDF: {len(results)}"
    ]
    
    gs = sum(r.gemini_success for r in results)
    qts = sum(r.qwen_thinking_success for r in results)
    qns = sum(r.qwen_no_thinking_success for r in results)
    
    lines += [
        f"- Gemini 成功: {gs}/{len(results)}",
        f"- Qwen (思考模式) 成功: {qts}/{len(results)}",
        f"- Qwen (非思考模式) 成功: {qns}/{len(results)}"
    ]
    
    # 平均耗时
    gt = [r.gemini_time for r in results if r.gemini_success and r.gemini_time > 0]
    qtt = [r.qwen_thinking_time for r in results if r.qwen_thinking_success and r.qwen_thinking_time > 0]
    qnt = [r.qwen_no_thinking_time for r in results if r.qwen_no_thinking_success and r.qwen_no_thinking_time > 0]
    
    if gt: lines.append(f"- Gemini 平均耗时: {sum(gt)/len(gt):.1f}s")
    if qtt: lines.append(f"- Qwen (思考) 平均耗时: {sum(qtt)/len(qtt):.1f}s")
    if qnt: lines.append(f"- Qwen (非思考) 平均耗时: {sum(qnt)/len(qnt):.1f}s")
    
    # Token 统计
    g_tokens = sum(r.gemini_tokens.get("total", 0) for r in results if r.gemini_success)
    qt_tokens = sum(r.qwen_thinking_tokens.get("total", 0) for r in results if r.qwen_thinking_success)
    qn_tokens = sum(r.qwen_no_thinking_tokens.get("total", 0) for r in results if r.qwen_no_thinking_success)
    
    lines += [
        "\n## 2. Token 消耗",
        f"- Gemini: {g_tokens:,} tokens",
        f"- Qwen (思考): {qt_tokens:,} tokens",
        f"- Qwen (非思考): {qn_tokens:,} tokens"
    ]
    
    # 评测指标
    valid = [r for r in results if r.metrics]
    if valid:
        lines.append("\n## 3. 评测指标对比")
        
        # 来源命中率
        lines.append("\n### 来源命中率")
        for name, key in [("Gemini", "gemini"), ("Qwen (思考)", "qwen_thinking"), ("Qwen (非思考)", "qwen_no_thinking")]:
            rates = [r.metrics.get(f"{key}_source_rate", 0) for r in valid if f"{key}_source_rate" in r.metrics]
            if rates:
                lines.append(f"- {name}: {sum(rates)/len(rates):.1%}")
        
        # Key Results 数量
        lines.append("\n### Key Results 提取数量")
        for name, key in [("Gemini", "gemini"), ("Qwen (思考)", "qwen_thinking"), ("Qwen (非思考)", "qwen_no_thinking")]:
            counts = [r.metrics.get(f"{key}_key_results_count", 0) for r in valid if f"{key}_key_results_count" in r.metrics]
            if counts:
                lines.append(f"- {name}: {sum(counts)/len(counts):.1f}")
    
    # 详细结果表
    lines += [
        "\n## 4. 详细结果",
        "\n| PDF | 页 | 字符 | Gemini | Q-思考 | Q-非思考 | KR数量 |",
        "|-----|---|------|--------|--------|----------|--------|"
    ]
    
    for r in results:
        gs = "✓" if r.gemini_success else "✗"
        qts = "✓" if r.qwen_thinking_success else "✗"
        qns = "✓" if r.qwen_no_thinking_success else "✗"
        
        kr = "-"
        if r.metrics:
            g_kr = r.metrics.get("gemini_key_results_count", "-")
            qt_kr = r.metrics.get("qwen_thinking_key_results_count", "-")
            qn_kr = r.metrics.get("qwen_no_thinking_key_results_count", "-")
            kr = f"G:{g_kr} T:{qt_kr} N:{qn_kr}"
        
        lines.append(f"| {r.pdf_name[:22]} | {r.pdf_pages} | {r.pdf_chars} | {gs} | {qts} | {qns} | {kr} |")
    
    # 错误
    errs = [(r.pdf_name, r.gemini_error, r.qwen_thinking_error, r.qwen_no_thinking_error) 
            for r in results if r.gemini_error or r.qwen_thinking_error or r.qwen_no_thinking_error]
    if errs:
        lines.append("\n## 5. 错误")
        for n, ge, qte, qne in errs:
            if ge: lines.append(f"- {n} (Gemini): {ge[:60]}")
            if qte: lines.append(f"- {n} (Q-思考): {qte[:60]}")
            if qne: lines.append(f"- {n} (Q-非思考): {qne[:60]}")
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "pdf_samples"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval_results_vllm"))
    ap.add_argument("--qwen-url", default="http://localhost:8000/v1", help="vLLM API URL")
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[:args.limit]
    
    print(f"PDF 数量: {len(pdfs)}")
    print(f"Qwen API: {args.qwen_url}")
    print(f"分块配置: max={MAX_CHUNK_CHARS} chars, overlap={OVERLAP_CHARS} chars")
    print("-" * 60)
    
    results = []
    for i, p in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {p.name}")
        results.append(process_single_pdf(p, output_dir, args.qwen_url, args.skip_gemini, args.skip_qwen))
    
    # 保存结果
    (output_dir / "eval_results.json").write_text(
        json.dumps([{k: v for k, v in vars(r).items()} for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    generate_report(results, output_dir / "eval_report.md")
    print("\n完成!")

if __name__ == "__main__":
    main()
