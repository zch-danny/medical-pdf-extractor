#!/usr/bin/env python
"""
Gemini 使用 v5.0 提示词分类测试
"""
import json, re, sys, time, os, argparse
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pdf_summarizer.utils.pdf_parser import PDFParser

# 使用 v5.0 提示词
PROMPT_V50 = Path("/root/提示词/分类器/classifier_prompt_v5.md").read_text(encoding="utf-8")

lock = threading.Lock()
progress = {"done": 0, "total": 0, "errors": 0}

def extract_pdf_text(pdf_path: str) -> Tuple[str, int]:
    parser = PDFParser(max_pages=50)
    full_text, metadata = parser.extract_text_from_file(pdf_path)
    full_text = re.sub(r"\[Page\s+(\d+)\]", r"[p\1]", full_text)
    return full_text[:12000], metadata.get("pages", 0)

def extract_json_object(text: str) -> Dict[str, Any]:
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
    return json.loads(text[start:end])

def call_gemini(text: str) -> Tuple[Dict, float]:
    from openai import OpenAI
    api_key = os.environ.get("GEMINI_API_KEY", "")
    api_domain = os.environ.get("GEMINI_API_DOMAIN", "api.bltcy.ai")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
    
    client = OpenAI(api_key=api_key, base_url=f"https://{api_domain}/v1")
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_V50 + text}],
        temperature=0.1,
        max_tokens=2048,
    )
    elapsed = time.time() - start
    result = extract_json_object(response.choices[0].message.content)
    return result, elapsed

def process_pdf(pdf_path: Path) -> Dict:
    result = {"pdf": pdf_path.name, "type": "", "conf": "", "time": 0.0, "status": "ok", "error": "", "raw": {}}
    try:
        text, _ = extract_pdf_text(str(pdf_path))
        classify_result, elapsed = call_gemini(text)
        result["type"] = classify_result.get("type", "ERROR")
        result["conf"] = classify_result.get("conf", "")
        result["time"] = elapsed
        result["raw"] = classify_result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:100]
        with lock: progress["errors"] += 1
    
    with lock:
        progress["done"] += 1
        if progress["done"] % 10 == 0 or progress["done"] == progress["total"]:
            print(f"进度: {progress['done']}/{progress['total']} | 类型: {result['type']}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/root/autodl-tmp/pdf_input/pdf_input_09-1125")
    parser.add_argument("--output", default="/root/autodl-tmp/gemini_v5_classification.json")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    pdf_files = sorted(Path(args.input_dir).glob("*.pdf"))[:args.limit]
    progress["total"] = len(pdf_files)
    
    print("=" * 60)
    print("Gemini 使用 v5.0 提示词分类测试")
    print("=" * 60)
    print(f"PDF: {len(pdf_files)} | Workers: {args.workers}")
    print("=" * 60)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        results = [f.result() for f in as_completed(futures)]
    
    results.sort(key=lambda x: x["pdf"])
    type_counts = {}
    for r in results:
        t = r["type"] if r["status"] == "ok" else "ERROR"
        type_counts[t] = type_counts.get(t, 0) + 1
    
    success = sum(1 for r in results if r["status"] == "ok")
    print(f"\n总耗时: {time.time()-start_time:.1f}s | 成功: {success}/{len(results)}")
    print("类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/len(results):.1f}%)")
    
    Path(args.output).write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "prompt": "v5.0",
        "type_distribution": type_counts,
        "results": results
    }, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {args.output}")

if __name__ == "__main__":
    main()
