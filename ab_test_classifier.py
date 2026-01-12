#!/usr/bin/env python
"""
A/B 测试脚本 - 比较分类器提示词 v4.3 vs v5.0
"""
import json, re, sys, time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

# 提示词路径
PROMPT_DIR = Path("/root/提示词/分类器")
PROMPT_V43 = (PROMPT_DIR / "classifier_prompt.md").read_text(encoding="utf-8")
PROMPT_V50 = (PROMPT_DIR / "classifier_prompt_v5.md").read_text(encoding="utf-8")

# 样本目录
SAMPLE_DIR = Path("/root/autodl-tmp/pdf_summarization_data/pdf_samples")
INPUT_DIR = Path("/root/autodl-tmp/pdf_input/pdf_input_09-1125")

@dataclass
class ABResult:
    pdf_name: str
    v43_type: str = ""
    v43_conf: str = ""
    v43_time: float = 0.0
    v50_type: str = ""
    v50_conf: str = ""
    v50_time: float = 0.0
    gemini_type: str = ""
    v43_match_gemini: bool = False
    v50_match_gemini: bool = False
    v43_raw: Dict = field(default_factory=dict)
    v50_raw: Dict = field(default_factory=dict)
    error: str = ""

def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    parser = PDFParser(max_pages=50)  # 分类只需前50页
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
    # 截取前12000字符用于分类
    return full_text[:12000], metadata.get("pages", 0)

def extract_json_object(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
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

def call_qwen(prompt: str, text: str, base_url: str = "http://localhost:8000/v1") -> Tuple[Dict, float]:
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    start = time.time()
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": prompt + text}],
        temperature=0.1,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    elapsed = time.time() - start
    
    result = extract_json_object(response.choices[0].message.content)
    return result, elapsed

def run_ab_test(pdf_files: list, gemini_truth: Dict[str, str], max_samples: int = 20):
    results = []
    
    for i, pdf_path in enumerate(pdf_files[:max_samples]):
        pdf_name = pdf_path.name
        print(f"\n[{i+1}/{min(len(pdf_files), max_samples)}] {pdf_name}")
        
        result = ABResult(pdf_name=pdf_name, gemini_type=gemini_truth.get(pdf_name, "UNKNOWN"))
        
        try:
            text, pages = extract_pdf_text(str(pdf_path))
            print(f"  文本长度: {len(text)} 字符, {pages} 页")
            
            # v4.3 测试
            print("  测试 v4.3...", end=" ", flush=True)
            v43_result, v43_time = call_qwen(PROMPT_V43, text)
            result.v43_type = v43_result.get("type", "ERROR")
            result.v43_conf = v43_result.get("conf", "")
            result.v43_time = v43_time
            result.v43_raw = v43_result
            print(f"{result.v43_type} ({v43_time:.1f}s)")
            
            # v5.0 测试
            print("  测试 v5.0...", end=" ", flush=True)
            v50_result, v50_time = call_qwen(PROMPT_V50, text)
            result.v50_type = v50_result.get("type", "ERROR")
            result.v50_conf = v50_result.get("conf", "")
            result.v50_time = v50_time
            result.v50_raw = v50_result
            print(f"{result.v50_type} ({v50_time:.1f}s)")
            
            # 对比
            result.v43_match_gemini = result.v43_type == result.gemini_type
            result.v50_match_gemini = result.v50_type == result.gemini_type
            
            match_info = []
            if result.gemini_type != "UNKNOWN":
                match_info.append(f"Gemini={result.gemini_type}")
                match_info.append(f"v4.3={'✓' if result.v43_match_gemini else '✗'}")
                match_info.append(f"v5.0={'✓' if result.v50_match_gemini else '✗'}")
            print(f"  对比: {' | '.join(match_info) if match_info else '无ground truth'}")
            
        except Exception as e:
            result.error = str(e)
            print(f"  错误: {e}")
        
        results.append(result)
    
    return results

def print_summary(results: list):
    print("\n" + "="*60)
    print("A/B 测试结果汇总")
    print("="*60)
    
    # 统计
    total = len(results)
    with_gt = [r for r in results if r.gemini_type != "UNKNOWN"]
    v43_matches = sum(1 for r in with_gt if r.v43_match_gemini)
    v50_matches = sum(1 for r in with_gt if r.v50_match_gemini)
    both_match = sum(1 for r in with_gt if r.v43_match_gemini and r.v50_match_gemini)
    v50_better = sum(1 for r in with_gt if r.v50_match_gemini and not r.v43_match_gemini)
    v43_better = sum(1 for r in with_gt if r.v43_match_gemini and not r.v50_match_gemini)
    
    print(f"\n样本总数: {total}")
    print(f"有 Gemini Ground Truth: {len(with_gt)}")
    
    if with_gt:
        print(f"\n与 Gemini 一致性:")
        print(f"  v4.3: {v43_matches}/{len(with_gt)} ({100*v43_matches/len(with_gt):.1f}%)")
        print(f"  v5.0: {v50_matches}/{len(with_gt)} ({100*v50_matches/len(with_gt):.1f}%)")
        print(f"\n改进分析:")
        print(f"  两者都正确: {both_match}")
        print(f"  v5.0 优于 v4.3: {v50_better}")
        print(f"  v4.3 优于 v5.0: {v43_better}")
    
    # 分歧案例
    disagreements = [r for r in results if r.v43_type != r.v50_type]
    if disagreements:
        print(f"\n分歧案例 ({len(disagreements)} 个):")
        for r in disagreements:
            gt = f" [GT={r.gemini_type}]" if r.gemini_type != "UNKNOWN" else ""
            print(f"  {r.pdf_name[:40]}... v4.3={r.v43_type} vs v5.0={r.v50_type}{gt}")
    
    # 时间统计
    v43_times = [r.v43_time for r in results if r.v43_time > 0]
    v50_times = [r.v50_time for r in results if r.v50_time > 0]
    if v43_times and v50_times:
        print(f"\n平均耗时:")
        print(f"  v4.3: {sum(v43_times)/len(v43_times):.2f}s")
        print(f"  v5.0: {sum(v50_times)/len(v50_times):.2f}s")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20, help="测试样本数")
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="PDF输入目录")
    parser.add_argument("--output", default="/root/autodl-tmp/ab_test_v43_vs_v50.json", help="结果输出")
    args = parser.parse_args()
    
    # 加载 Gemini ground truth
    gt_file = Path("/root/autodl-tmp/gemini_ground_truth.json")
    gemini_truth = json.loads(gt_file.read_text()) if gt_file.exists() else {}
    
    # 获取PDF文件列表
    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))[:args.samples]
    
    if not pdf_files:
        print(f"未找到 PDF 文件: {input_dir}")
        return
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件，测试前 {args.samples} 个")
    print(f"Gemini Ground Truth: {len(gemini_truth)} 条")
    
    # 运行测试
    results = run_ab_test(pdf_files, gemini_truth, args.samples)
    
    # 打印汇总
    print_summary(results)
    
    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "samples": args.samples,
        "results": [
            {
                "pdf": r.pdf_name,
                "v43": {"type": r.v43_type, "conf": r.v43_conf, "time": r.v43_time, "raw": r.v43_raw},
                "v50": {"type": r.v50_type, "conf": r.v50_conf, "time": r.v50_time, "raw": r.v50_raw},
                "gemini": r.gemini_type,
                "v43_match": r.v43_match_gemini,
                "v50_match": r.v50_match_gemini,
                "error": r.error
            }
            for r in results
        ]
    }
    
    Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {args.output}")

if __name__ == "__main__":
    main()
