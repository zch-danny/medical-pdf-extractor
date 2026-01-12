#!/usr/bin/env python
"""
批量分类脚本 - 使用 v5.0 提示词
"""
import json, re, sys, time, argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

# 提示词
PROMPT_V50 = (Path("/root/提示词/分类器/classifier_prompt_v5.md")).read_text(encoding="utf-8")

# 线程安全计数器
lock = threading.Lock()
progress = {"done": 0, "total": 0, "errors": 0}

def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    parser = PDFParser(max_pages=50)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
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

def call_qwen(text: str, base_url: str = "http://localhost:8000/v1") -> Tuple[Dict, float, int]:
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    
    start = time.time()
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": PROMPT_V50 + text}],
        temperature=0.1,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    elapsed = time.time() - start
    tokens = response.usage.total_tokens
    
    result = extract_json_object(response.choices[0].message.content)
    return result, elapsed, tokens

def process_pdf(pdf_path: Path) -> Dict:
    result = {
        "pdf": pdf_path.name,
        "type": "",
        "conf": "",
        "time": 0.0,
        "tokens": 0,
        "status": "ok",
        "error": "",
        "raw": {}
    }
    
    try:
        text, pages = extract_pdf_text(str(pdf_path))
        classify_result, elapsed, tokens = call_qwen(text)
        
        result["type"] = classify_result.get("type", "ERROR")
        result["conf"] = classify_result.get("conf", "")
        result["time"] = elapsed
        result["tokens"] = tokens
        result["raw"] = classify_result
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        with lock:
            progress["errors"] += 1
    
    with lock:
        progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        errors = progress["errors"]
        if done % 10 == 0 or done == total:
            print(f"进度: {done}/{total} ({100*done/total:.1f}%) | 错误: {errors} | 当前: {result['type']}")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/root/autodl-tmp/pdf_input/pdf_input_09-1125")
    parser.add_argument("--output", default="/root/autodl-tmp/v5_classification_all.json")
    parser.add_argument("--workers", type=int, default=5, help="并发数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量，0=全部")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))
    
    if args.limit > 0:
        pdf_files = pdf_files[:args.limit]
    
    progress["total"] = len(pdf_files)
    
    print(f"=" * 60)
    print(f"v5.0 批量分类测试")
    print(f"=" * 60)
    print(f"PDF 数量: {len(pdf_files)}")
    print(f"并发数: {args.workers}")
    print(f"输出文件: {args.output}")
    print(f"=" * 60)
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # 按文件名排序
    results.sort(key=lambda x: x["pdf"])
    
    # 统计
    type_counts = {}
    for r in results:
        t = r["type"] if r["status"] == "ok" else "ERROR"
        type_counts[t] = type_counts.get(t, 0) + 1
    
    success_count = sum(1 for r in results if r["status"] == "ok")
    total_tokens = sum(r["tokens"] for r in results)
    avg_time = sum(r["time"] for r in results if r["time"] > 0) / max(success_count, 1)
    
    print(f"\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"成功: {success_count}/{len(results)}")
    print(f"平均耗时: {avg_time:.2f}s/文件")
    print(f"总 tokens: {total_tokens:,}")
    print(f"\n类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/len(results):.1f}%)")
    
    # 保存结果
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": "v5.0",
        "total_files": len(pdf_files),
        "success_count": success_count,
        "total_time_seconds": total_time,
        "total_tokens": total_tokens,
        "type_distribution": type_counts,
        "results": results
    }
    
    Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {args.output}")

if __name__ == "__main__":
    main()
