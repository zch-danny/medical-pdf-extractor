#!/usr/bin/env python
"""
Gemini 中立分类测试 - 使用通用提示词（非定制版本）
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

# 通用分类提示词 - 不使用任何定制规则
NEUTRAL_PROMPT = """你是一位医学文献分类专家。请根据以下文献内容，判断其类型。

## 文献类型定义

1. **GUIDELINE** - 临床实践指南/专家共识/行业标准
   - 由权威医学组织发布
   - 包含临床推荐建议
   - 用于指导临床决策

2. **RCT** - 随机对照试验
   - 随机分组
   - 有干预组和对照组
   - 报告临床结局

3. **META** - Meta分析/系统评价
   - 系统性文献检索
   - 对多项研究进行定量或定性综合

4. **COHORT** - 队列研究
   - 前瞻性或回顾性随访
   - 比较不同暴露组的结局

5. **CASE_CONTROL** - 病例对照研究
   - 从结局出发
   - 回顾比较病例组与对照组的暴露差异

6. **CROSS_SECTIONAL** - 横断面研究
   - 单一时间点调查
   - 报告患病率或现状

7. **REVIEW** - 综述/述评
   - 对某主题的文献回顾和总结
   - 包括叙述性综述、专家述评、文献解读

8. **OTHER** - 其他类型
   - 病例报告、方法学文章、社论等

## 输出要求

只输出 JSON，格式如下：
```json
{
  "type": "类型",
  "confidence": "high|medium|low",
  "reason": "简要说明判定理由"
}
```

## 待分类文献

"""

# 线程安全
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
        messages=[{"role": "user", "content": NEUTRAL_PROMPT + text}],
        temperature=0.1,
        max_tokens=1024,
    )
    elapsed = time.time() - start
    
    result = extract_json_object(response.choices[0].message.content)
    return result, elapsed

def process_pdf(pdf_path: Path) -> Dict:
    result = {
        "pdf": pdf_path.name,
        "type": "",
        "confidence": "",
        "reason": "",
        "time": 0.0,
        "status": "ok",
        "error": ""
    }
    
    try:
        text, pages = extract_pdf_text(str(pdf_path))
        classify_result, elapsed = call_gemini(text)
        
        result["type"] = classify_result.get("type", "ERROR")
        result["confidence"] = classify_result.get("confidence", "")
        result["reason"] = classify_result.get("reason", "")[:200]
        result["time"] = elapsed
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:100]
        with lock:
            progress["errors"] += 1
    
    with lock:
        progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        if done % 10 == 0 or done == total:
            print(f"进度: {done}/{total} ({100*done/total:.1f}%) | 类型: {result['type']}")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/root/autodl-tmp/pdf_input/pdf_input_09-1125")
    parser.add_argument("--output", default="/root/autodl-tmp/gemini_neutral_classification.json")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    parser.add_argument("--limit", type=int, default=50, help="测试样本数")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))[:args.limit]
    
    progress["total"] = len(pdf_files)
    
    print("=" * 60)
    print("Gemini 中立分类测试 (通用提示词)")
    print("=" * 60)
    print(f"PDF 数量: {len(pdf_files)}")
    print(f"并发数: {args.workers}")
    print(f"模型: {os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash-preview-05-20')}")
    print("=" * 60)
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    results.sort(key=lambda x: x["pdf"])
    
    # 统计
    type_counts = {}
    for r in results:
        t = r["type"] if r["status"] == "ok" else "ERROR"
        type_counts[t] = type_counts.get(t, 0) + 1
    
    success = sum(1 for r in results if r["status"] == "ok")
    
    print(f"\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总耗时: {total_time:.1f}s")
    print(f"成功: {success}/{len(results)}")
    print(f"\n类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/len(results):.1f}%)")
    
    # 保存
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt": "neutral (通用提示词)",
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20"),
        "total_files": len(pdf_files),
        "success_count": success,
        "type_distribution": type_counts,
        "results": results
    }
    
    Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {args.output}")

if __name__ == "__main__":
    main()
