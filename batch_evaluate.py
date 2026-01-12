#!/usr/bin/env python
"""批量处理 PDF 并对比 Gemini vs Qwen 结果 - 支持 OpenAI 兼容 API"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# 加载 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

@dataclass
class EvalResult:
    pdf_name: str
    pdf_pages: int
    pdf_chars: int
    gemini_time: float = 0.0
    qwen_time: float = 0.0
    gemini_success: bool = False
    qwen_success: bool = False
    gemini_error: str = ""
    qwen_error: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

def extract_pdf_text(pdf_path: str, max_pages: int = 200) -> Tuple[str, Dict[str, Any]]:
    parser = PDFParser(max_pages=max_pages)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
    return full_text, metadata

def get_page_markers(text: str) -> List[str]:
    return re.findall(r"\[p(\d+)\]", text)

GEMINI_PROMPT = """你是一名"循证医学 + 临床研究方法学"的文献分析助手。请基于输入的医学文献内容，生成【可评测的结构化摘要 + 证据提取】。

文本中已按页插入页码标记，例如：[p12]。你必须把这些页码标记用于来源定位。

硬性规则：
- 只能基于提供的文本，不得编造；不确定就写 "未在文本中找到"。
- 所有涉及数值的条目必须带来源页码 sources（数组），例如 ["p12","p13"]。
- 数值必须原样保留。
- 输出必须是严格 JSON（不要 Markdown）。

输出 JSON 结构：
{
  "doc_metadata": {"title": "", "year": "", "journal_or_source": "", "doi_or_trial_id": "", "funding_and_coi": {"funding": "", "conflicts": "", "sources": []}},
  "study_type": {"primary_type": "RCT/队列/病例对照/横断面/系统综述与Meta/诊断试验/指南/其他/unknown", "setting_and_design": "", "registration": "", "sources": []},
  "pico": {
    "population": {"summary": "", "inclusion": [], "exclusion": [], "baseline": [], "n": "", "sources": []},
    "intervention": {"summary": "", "details": [], "sources": []},
    "comparison": {"summary": "", "details": [], "sources": []},
    "outcomes": {"primary": [], "secondary": [], "sources": []}
  },
  "methods": {
    "randomization_and_blinding": {"randomization": "", "blinding": "", "allocation": "", "sources": []},
    "follow_up": {"duration": "", "loss_to_follow_up": "", "sources": []},
    "analysis": {"analysis_set": "ITT/PP/其他/未报告", "stats_methods": "", "adjustment": "", "sources": []}
  },
  "key_results": [{"outcome": "", "timepoint": "", "measure": "", "value": "", "ci_95": "", "p_value": "", "groups": "", "direction": "", "notes": "", "sources": []}],
  "safety": {"common_adverse_events": [], "serious_adverse_events": [], "notes": "", "sources": []},
  "limitations": [{"item": "", "sources": []}],
  "clinical_implications": [{"item": "", "sources": []}],
  "conclusions": [{"item": "", "sources": []}]
}

【文献内容】
"""

def call_gemini_openai_compat(text: str, api_key: str, api_domain: str, model: str) -> Dict[str, Any]:
    from openai import OpenAI
    base_url = f"https://{api_domain}/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = GEMINI_PROMPT + text
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=8192,
    )
    output = response.choices[0].message.content
    return extract_json_object(output)

def run_qwen_mapreduce(pdf_path: str, model_name: str, max_input_tokens: int = 4000) -> Dict[str, Any]:
    import subprocess
    out_path = Path(pdf_path).with_suffix(".qwen.json")
    cmd = ["python", str(PROJECT_ROOT / "medical_structured_mapreduce.py"), str(pdf_path),
           "--model", model_name, "--max-input-tokens", str(max_input_tokens), "--out", str(out_path)]
    env = os.environ.copy()
    env["HF_HOME"] = "/root/autodl-tmp/hf_cache"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"Qwen failed: {result.stderr[-500:]}")
    return json.loads(out_path.read_text(encoding="utf-8"))

def extract_json_object(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m: text = m.group(1)
    start = text.find("{")
    if start < 0: raise ValueError(f"No JSON: {text[:200]}")
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
    if end is None: raise ValueError(f"Unclosed JSON: {text[:200]}")
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

def count_non_empty_fields(data, fields):
    non_empty = 0
    for f in fields:
        v = data
        for p in f.split("."): v = v.get(p, "") if isinstance(v, dict) else ""
        if v and v not in ("unknown", "未报告", "not_found"): non_empty += 1
    return non_empty, len(fields)

def compare_key_results(gemini, qwen):
    g_kr, q_kr = gemini.get("key_results", []) or [], qwen.get("key_results", []) or []
    g_out = set(kr.get("outcome", "") for kr in g_kr if isinstance(kr, dict))
    q_out = set(kr.get("outcome", "") for kr in q_kr if isinstance(kr, dict))
    return {"gemini_count": len(g_kr), "qwen_count": len(q_kr), "common_outcomes": len(g_out & q_out),
            "only_gemini": list(g_out - q_out)[:5], "only_qwen": list(q_out - g_out)[:5]}

def evaluate(gemini, qwen, valid_pages):
    g_h, g_t, g_r = compute_source_hit_rate(gemini, valid_pages)
    q_h, q_t, q_r = compute_source_hit_rate(qwen, valid_pages)
    kr = compare_key_results(gemini, qwen)
    fields = ["doc_metadata.title", "doc_metadata.year", "study_type.primary_type", "pico.population.summary", "pico.intervention.summary"]
    g_f, g_tf = count_non_empty_fields(gemini, fields)
    q_f, q_tf = count_non_empty_fields(qwen, fields)
    return {
        "source_hit_rate": {"gemini": {"hits": g_h, "total": g_t, "rate": round(g_r, 3)}, "qwen": {"hits": q_h, "total": q_t, "rate": round(q_r, 3)}},
        "key_results_comparison": kr,
        "field_completion": {"gemini": {"filled": g_f, "total": g_tf, "rate": round(g_f/g_tf, 3) if g_tf else 0}, "qwen": {"filled": q_f, "total": q_tf, "rate": round(q_f/q_tf, 3) if q_tf else 0}}
    }

def process_single_pdf(pdf_path, api_key, api_domain, model, qwen_model, output_dir, skip_gemini=False, skip_qwen=False):
    result = EvalResult(pdf_name=pdf_path.name, pdf_pages=0, pdf_chars=0)
    try:
        full_text, metadata = extract_pdf_text(str(pdf_path))
        result.pdf_pages, result.pdf_chars = metadata.get("processed_pages", 0), metadata.get("char_count", 0)
        valid_pages = get_page_markers(full_text)
    except Exception as e:
        result.gemini_error = result.qwen_error = f"PDF parse failed: {e}"
        return result
    if result.pdf_chars < 500:
        result.gemini_error = result.qwen_error = "PDF too short or invalid"
        return result
    gemini_path, qwen_path = output_dir / f"{pdf_path.stem}.gemini.json", output_dir / f"{pdf_path.stem}.qwen.json"
    
    if not skip_gemini:
        try:
            t0 = time.time()
            gemini_result = call_gemini_openai_compat(full_text, api_key, api_domain, model)
            result.gemini_time, result.gemini_success = time.time() - t0, True
            gemini_path.write_text(json.dumps(gemini_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Gemini: {result.gemini_time:.1f}s ✓")
        except Exception as e:
            result.gemini_error = str(e)[:200]
            print(f"  Gemini: FAILED - {str(e)[:100]}")
    elif gemini_path.exists():
        result.gemini_success = True
        print(f"  Gemini: (cached) ✓")
    
    if not skip_qwen:
        try:
            t0 = time.time()
            run_qwen_mapreduce(str(pdf_path), qwen_model)
            result.qwen_time, result.qwen_success = time.time() - t0, True
            src = pdf_path.with_suffix(".structured.json")
            if src.exists(): qwen_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Qwen: {result.qwen_time:.1f}s ✓")
        except Exception as e:
            result.qwen_error = str(e)[:200]
            print(f"  Qwen: FAILED - {str(e)[:100]}")
    elif qwen_path.exists():
        result.qwen_success = True
        print(f"  Qwen: (cached) ✓")
    
    if result.gemini_success and result.qwen_success:
        result.metrics = evaluate(json.loads(gemini_path.read_text()), json.loads(qwen_path.read_text()), valid_pages)
    return result

def generate_report(results, output_path):
    lines = ["# 医学文献 PDF 摘要评测报告", f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "\n## 1. 总览", f"\n处理 PDF: {len(results)}"]
    gs, qs = sum(r.gemini_success for r in results), sum(r.qwen_success for r in results)
    lines += [f"- Gemini 成功: {gs}/{len(results)}", f"- Qwen 成功: {qs}/{len(results)}", f"- 双方成功: {sum(r.gemini_success and r.qwen_success for r in results)}/{len(results)}"]
    gt = [r.gemini_time for r in results if r.gemini_success and r.gemini_time > 0]
    qt = [r.qwen_time for r in results if r.qwen_success and r.qwen_time > 0]
    if gt: lines.append(f"- Gemini 平均耗时: {sum(gt)/len(gt):.1f}s")
    if qt: lines.append(f"- Qwen 平均耗时: {sum(qt)/len(qt):.1f}s")
    
    valid = [r for r in results if r.metrics]
    if valid:
        lines.append("\n## 2. 评测指标")
        gr = [r.metrics["source_hit_rate"]["gemini"]["rate"] for r in valid]
        qr = [r.metrics["source_hit_rate"]["qwen"]["rate"] for r in valid]
        lines += ["\n### 来源命中率", f"- Gemini: {sum(gr)/len(gr):.1%}", f"- Qwen: {sum(qr)/len(qr):.1%}"]
        gk = [r.metrics["key_results_comparison"]["gemini_count"] for r in valid]
        qk = [r.metrics["key_results_comparison"]["qwen_count"] for r in valid]
        lines += ["\n### Key Results 数量", f"- Gemini: {sum(gk)/len(gk):.1f}", f"- Qwen: {sum(qk)/len(qk):.1f}"]
    
    lines += ["\n## 3. 详细结果", "\n| PDF | 页 | 字符 | Gemini | Qwen | 来源命中 | KR |", "|-----|---|------|--------|------|---------|---|"]
    for r in results:
        gs, qs = "✓" if r.gemini_success else "✗", "✓" if r.qwen_success else "✗"
        if r.metrics:
            src = f"G:{r.metrics['source_hit_rate']['gemini']['rate']:.0%} Q:{r.metrics['source_hit_rate']['qwen']['rate']:.0%}"
            kr = f"G:{r.metrics['key_results_comparison']['gemini_count']} Q:{r.metrics['key_results_comparison']['qwen_count']}"
        else: src, kr = "-", "-"
        lines.append(f"| {r.pdf_name[:25]} | {r.pdf_pages} | {r.pdf_chars} | {gs} | {qs} | {src} | {kr} |")
    
    errs = [(r.pdf_name, r.gemini_error, r.qwen_error) for r in results if r.gemini_error or r.qwen_error]
    if errs:
        lines.append("\n## 4. 错误")
        for n, ge, qe in errs:
            if ge: lines.append(f"- {n} (G): {ge[:80]}")
            if qe: lines.append(f"- {n} (Q): {qe[:80]}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(PROJECT_ROOT / "pdf_samples"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval_results"))
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    ap.add_argument("--api-domain", default=os.environ.get("GEMINI_API_DOMAIN", "api.bltcy.ai"))
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"))
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--skip-gemini", action="store_true")
    ap.add_argument("--skip-qwen", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    
    pdf_dir, output_dir = Path(args.pdf_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))[:args.limit] if args.limit else sorted(pdf_dir.glob("*.pdf"))
    
    print(f"PDF: {len(pdfs)} | Domain: {args.api_domain} | Model: {args.model} | Key: {'✓' if args.api_key else '✗'}")
    print("-" * 60)
    
    results = []
    for i, p in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {p.name}")
        results.append(process_single_pdf(p, args.api_key, args.api_domain, args.model, args.qwen_model, output_dir, args.skip_gemini, args.skip_qwen))
    
    (output_dir / "eval_results.json").write_text(json.dumps([vars(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    generate_report(results, output_dir / "eval_report.md")
    print("\n完成!")

if __name__ == "__main__":
    main()
