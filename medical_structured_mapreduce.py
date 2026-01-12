#!/usr/bin/env python
"""医学文献 PDF：结构感知 Map-Reduce 高精度提取（本地 Qwen3-8B INT8）

特点：
- PDF 文本抽取带页码标记（[p12]），用于来源定位
- 结构路由：Design / Evidence / Conclusion
- 分治提取：不同 section 用不同 Prompt，降低认知负荷，提高数值准确性
- 合并阶段：用 Python 代码合并 JSON，而非再次调用 LLM（减少幻觉与覆盖风险）

默认使用本地 transformers + bitsandbytes INT8 加载 Qwen3-8B。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 让脚本可直接在项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pdf_summarizer.utils.pdf_parser import PDFParser
from pdf_summarizer.utils.medical_structure_router import route_medical_sections


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


PROMPT_A = r"""你是一名临床研究方法学专家。请基于以下【方法学/背景片段】提取研究设计信息。

【输入文本】
{chunk_a_content}

【输出要求】
- 只输出严格 JSON（不要 Markdown，不要解释）。
- 必须引用页码：把来源页码写入 sources，例如 ["p12","p13"]。
- 如果没找到，填 "not_found" 或空数组。
- 不要输出 outcomes 的数值（效应量/CI/p 值放到 Results 提取里）。

【输出 JSON】（仅包含以下字段，字段名必须一致）
{{
  "doc_metadata": {{
    "title": "",
    "year": "",
    "journal_or_source": "",
    "doi_or_trial_id": "",
    "funding_and_coi": {{"funding": "", "conflicts": "", "sources": []}}
  }},
  "study_type": {{
    "primary_type": "RCT/队列/病例对照/横断面/系统综述与Meta/诊断试验/指南/其他/unknown",
    "setting_and_design": "",
    "registration": "",
    "sources": []
  }},
  "pico": {{
    "population": {{"summary": "", "inclusion": [], "exclusion": [], "baseline": [], "n": "", "sources": []}},
    "intervention": {{"summary": "", "details": [], "sources": []}},
    "comparison": {{"summary": "", "details": [], "sources": []}},
    "outcomes": {{"primary": [], "secondary": [], "sources": []}}
  }},
  "methods": {{
    "randomization_and_blinding": {{"randomization": "", "blinding": "", "allocation": "", "sources": []}},
    "follow_up": {{"duration": "", "loss_to_follow_up": "", "sources": []}},
    "analysis": {{"analysis_set": "ITT/PP/其他/未报告", "stats_methods": "", "adjustment": "", "sources": []}}
  }}
}}
"""


PROMPT_B = r"""你是一名医学数据录入员。你的唯一任务是从【结果/证据片段】中精准提取数值（不要总结、不要解释）。

【输入文本】
{chunk_b_content}

【硬性约束】
- 只输出严格 JSON（不要 Markdown，不要解释）。
- 严禁进行数学计算，必须逐字摘录原文数字。
- 必须引用页码 sources，例如 ["p12"]。
- 如果同一指标在表格与正文重复，优先表格。

【输出 JSON】（仅包含以下字段，字段名必须一致）
{{
  "key_results": [
    {{
      "outcome": "",
      "timepoint": "",
      "measure": "RR/OR/HR/MD/SMD/AUC/敏感度/特异度/其他/未报告",
      "value": "",
      "ci_95": "",
      "p_value": "",
      "groups": "干预 vs 对照（或分组说明，尽量包含n）",
      "direction": "benefit/harm/null/unknown",
      "notes": "",
      "sources": []
    }}
  ],
  "safety": {{
    "common_adverse_events": [{{"event": "", "rate": "", "groups": "", "sources": []}}],
    "serious_adverse_events": [{{"event": "", "rate": "", "groups": "", "sources": []}}],
    "notes": "",
    "sources": []
  }}
}}
"""


PROMPT_C = r"""你是一名医学期刊审稿人。请基于以下【讨论/结论片段】，提取作者明确陈述的局限性、临床意义与结论（不要添加外部观点）。

【输入文本】
{chunk_c_content}

【输出要求】
- 只输出严格 JSON（不要 Markdown，不要解释）。
- 每条条目必须带 sources（页码）。

【输出 JSON】（仅包含以下字段，字段名必须一致）
{{
  "limitations": [{{"item": "", "sources": []}}],
  "clinical_implications": [{{"item": "", "sources": []}}],
  "conclusions": [{{"item": "", "sources": []}}]
}}
"""


def empty_final_schema() -> Dict[str, Any]:
    return {
        "doc_metadata": {
            "title": "",
            "year": "",
            "journal_or_source": "",
            "doi_or_trial_id": "",
            "funding_and_coi": {"funding": "", "conflicts": "", "sources": []},
        },
        "study_type": {
            "primary_type": "unknown",
            "setting_and_design": "",
            "registration": "",
            "sources": [],
        },
        "pico": {
            "population": {
                "summary": "",
                "inclusion": [],
                "exclusion": [],
                "baseline": [],
                "n": "",
                "sources": [],
            },
            "intervention": {"summary": "", "details": [], "sources": []},
            "comparison": {"summary": "", "details": [], "sources": []},
            "outcomes": {"primary": [], "secondary": [], "sources": []},
        },
        "methods": {
            "randomization_and_blinding": {
                "randomization": "",
                "blinding": "",
                "allocation": "",
                "sources": [],
            },
            "follow_up": {"duration": "", "loss_to_follow_up": "", "sources": []},
            "analysis": {
                "analysis_set": "未报告",
                "stats_methods": "",
                "adjustment": "",
                "sources": [],
            },
        },
        "key_results": [],
        "safety": {
            "common_adverse_events": [],
            "serious_adverse_events": [],
            "notes": "",
            "sources": [],
        },
        "limitations": [],
        "clinical_implications": [],
        "conclusions": [],
        "conflicts": [],
        "_warnings": [],
        "_debug": {},
    }


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def normalize_page_markers(text: str) -> str:
    # 支持旧格式 [Page 12] -> [p12]
    return re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", text)


def extract_pdf_text(pdf_path: str, max_pages: int) -> Tuple[str, Dict[str, Any]]:
    parser = PDFParser(max_pages=max_pages)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = normalize_page_markers(full_text)
    return full_text, metadata


def split_by_pages(text: str) -> List[str]:
    # 以页码标记为边界，尽量保持表格不被切碎（至少不跨页切）
    parts = re.split(r"(?=\[p\d+\])", text)
    return [p.strip() for p in parts if p.strip()]


def count_tokens(tokenizer, s: str) -> int:
    return len(tokenizer.encode(s, add_special_tokens=False))


def pack_pages_to_segments(pages: List[str], tokenizer, max_input_tokens: int) -> List[str]:
    segments: List[str] = []
    cur = ""
    for p in pages:
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if cur and count_tokens(tokenizer, candidate) > max_input_tokens:
            segments.append(cur.strip())
            cur = p
        else:
            cur = candidate
    if cur.strip():
        segments.append(cur.strip())
    return segments


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取首个 JSON object。"""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in output: {text[:200]!r}")

    in_str = False
    esc = False
    depth = 0
    end = None

    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        raise ValueError(f"Unclosed JSON object in output: {text[:200]!r}")

    raw = text[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 轻量修复：去掉可能的尾随逗号
        raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(raw2)


def load_local_qwen_int8(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    qconf = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconf,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    # Qwen3 支持 /no_think 标记来禁用思考模式，或者用 chat template
    # 这里用 chat template 包装，更稳定
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3 特有参数，禁用思考
    )
    
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def map_extract_design(model, tokenizer, text: str) -> Dict[str, Any]:
    out = generate(model, tokenizer, PROMPT_A.format(chunk_a_content=text), max_new_tokens=1536)
    return extract_json_object(out)


def map_extract_evidence(model, tokenizer, text: str) -> Dict[str, Any]:
    out = generate(model, tokenizer, PROMPT_B.format(chunk_b_content=text), max_new_tokens=4096)
    return extract_json_object(out)


def map_extract_conclusion(model, tokenizer, text: str) -> Dict[str, Any]:
    out = generate(model, tokenizer, PROMPT_C.format(chunk_c_content=text), max_new_tokens=1024)
    return extract_json_object(out)


def _parse_first_int(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\b(\d{2,6})\b", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def reduce_merge(
    a: Dict[str, Any],
    b_list: List[Dict[str, Any]],
    c: Dict[str, Any],
    router_debug: Dict[str, Any],
    pdf_meta: Dict[str, Any],
) -> Dict[str, Any]:
    final = empty_final_schema()

    # 先填充 A / C
    deep_update(final, {k: v for k, v in a.items() if k in ("doc_metadata", "study_type", "pico", "methods")})
    deep_update(final, {k: v for k, v in c.items() if k in ("limitations", "clinical_implications", "conclusions")})

    # 合并所有 B 分段的 key_results/safety
    key_results: List[Dict[str, Any]] = []
    common_ae: List[Dict[str, Any]] = []
    serious_ae: List[Dict[str, Any]] = []
    safety_notes: List[str] = []
    safety_sources: List[str] = []

    for b in b_list:
        for kr in b.get("key_results", []) or []:
            if isinstance(kr, dict):
                key_results.append(kr)
        s = b.get("safety") or {}
        for ae in s.get("common_adverse_events", []) or []:
            if isinstance(ae, dict):
                common_ae.append(ae)
        for ae in s.get("serious_adverse_events", []) or []:
            if isinstance(ae, dict):
                serious_ae.append(ae)
        if isinstance(s.get("notes"), str) and s.get("notes").strip():
            safety_notes.append(s.get("notes").strip())
        for src in s.get("sources", []) or []:
            if isinstance(src, str):
                safety_sources.append(src)

    final["key_results"] = key_results
    final["safety"]["common_adverse_events"] = common_ae
    final["safety"]["serious_adverse_events"] = serious_ae
    final["safety"]["notes"] = "\n".join(dict.fromkeys(safety_notes))
    final["safety"]["sources"] = sorted(set(safety_sources))

    # 简单冲突检测：A 的样本量 vs B 中 groups 字段提到的 n（若可解析）
    pop_n = _parse_first_int(str(final.get("pico", {}).get("population", {}).get("n", "")))
    if pop_n is not None and key_results:
        # 从首条 key_result 的 groups 中解析所有 n
        groups = str(key_results[0].get("groups", ""))
        nums = [int(x) for x in re.findall(r"\b(\d{2,6})\b", groups)[:4]]
        if nums:
            # 常见是两组 n
            est_total = sum(nums) if len(nums) <= 3 else None
            if est_total and abs(est_total - pop_n) / max(pop_n, 1) > 0.05:
                final["_warnings"].append(
                    {
                        "type": "sample_size_mismatch",
                        "population_n": pop_n,
                        "groups_n_sum": est_total,
                        "note": "Population n 与 Results 分组 n 之和差异>5%，建议人工核查。",
                    }
                )

    final["_debug"] = {
        "router": router_debug,
        "pdf_metadata": pdf_meta,
        "segments": {"evidence_segments": len(b_list)},
    }

    return final


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="PDF 文件路径")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF 模型名（默认 Qwen/Qwen3-8B）")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--max-input-tokens", type=int, default=8000, help="Evidence 单段最大输入 tokens（按页打包）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认同目录下生成）")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_path = Path(args.out) if args.out else pdf_path.with_suffix(".structured.json")

    t0 = time.time()
    print(f">> extracting text: {pdf_path}")
    full_text, pdf_meta = extract_pdf_text(str(pdf_path), max_pages=args.max_pages)

    print(">> routing sections...")
    routed = route_medical_sections(full_text)
    print(f"   router_debug={routed.debug}")

    print(">> loading model (INT8)...")
    tokenizer, model = load_local_qwen_int8(args.model)

    # Design / Conclusion 一次提取
    print(">> map A (design)...")
    a = map_extract_design(model, tokenizer, routed.design[:])

    print(">> map C (conclusion)...")
    c = map_extract_conclusion(model, tokenizer, routed.conclusion[:])

    # Evidence 按页打包成多段
    print(">> splitting evidence by pages...")
    pages = split_by_pages(routed.evidence)
    segments = pack_pages_to_segments(pages, tokenizer, max_input_tokens=args.max_input_tokens)
    print(f"   evidence_pages={len(pages)}, segments={len(segments)}")

    b_list: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, start=1):
        print(f">> map B (evidence) segment {i}/{len(segments)}...")
        b = map_extract_evidence(model, tokenizer, seg)
        b_list.append(b)

    print(">> reduce merge...")
    final = reduce_merge(a, b_list, c, routed.debug, pdf_meta)

    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    dt = time.time() - t0
    print(f">> done: {out_path} (elapsed {dt:.1f}s)")


if __name__ == "__main__":
    main()
