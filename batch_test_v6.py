#!/usr/bin/env python3
"""
批量测试 - 使用优化后的提取器v5/v4，测试10个PDF
"""
import json, time, requests, fitz, shutil, re
from pathlib import Path
from datetime import datetime

QWEN_API_URL = "http://localhost:8000/v1/chat/completions"
GPT_API_URL = "https://api.bltcy.ai/v1/chat/completions"
GPT_API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUTPUT_ROOT = Path('/root/extraction_test_results')
NUM_TESTS = 10  # 测试10个

# 使用最新版提取器
EXTRACTOR_MAP = {
    'GUIDELINE': 'GUIDELINE_extractor_v5.md',
    'META': 'META_extractor_v2.md',
    'REVIEW': 'REVIEW_extractor.md',
    'OTHER': 'OTHER_extractor_v4.md'  # 更新为v4
}

def extract_pdf_text(pdf_path, max_pages=5, max_chars=8000):
    doc = fitz.open(pdf_path)
    text_parts = []
    for i, page in enumerate(doc):
        if i >= max_pages: break
        text_parts.append(f"\n[第{i+1}页]\n" + page.get_text())
    doc.close()
    return ''.join(text_parts)[:max_chars]

def parse_json(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try: return json.loads(m.group())
        except: return None
    return None

def call_qwen(prompt, max_tokens=4000):
    payload = {"model": "qwen3-8b", "messages": [{"role": "user", "content": prompt}], 
               "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    start = time.time()
    r = requests.post(QWEN_API_URL, json=payload, timeout=180)
    result = r.json()
    return result['choices'][0]['message']['content'], time.time() - start, result.get('usage', {})

def call_gpt52(prompt, max_tokens=4000):
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], 
               "max_tokens": max_tokens, "temperature": 0.3}
    start = time.time()
    try:
        r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=120)
        result = r.json()
        if "error" in result: return f"错误: {result['error']}", time.time() - start, {}
        return result['choices'][0]['message']['content'], time.time() - start, result.get('usage', {})
    except Exception as e:
        return f"异常: {e}", time.time() - start, {}

def process_pdf(pdf_path, output_dir, idx, total):
    print(f"\n[{idx}/{total}] {pdf_path.name}")
    
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, pdf_output_dir / pdf_path.name)
    
    pdf_text = extract_pdf_text(pdf_path)
    result = {"filename": pdf_path.name, "test_time": datetime.now().isoformat()}
    
    # 1. 分类
    classifier_prompt = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
    classify_content, classify_time, classify_usage = call_qwen(
        classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}"), 2000)
    classify_json = parse_json(classify_content)
    doc_type = classify_json.get('type', 'OTHER') if classify_json else 'OTHER'
    print(f"  分类: {doc_type} ({classify_time:.1f}s)")
    result["classify"] = {"type": doc_type, "time": classify_time, "tokens": classify_usage.get('total_tokens', 0)}
    
    # 2. 提取
    extractor_file = EXTRACTOR_MAP.get(doc_type, 'OTHER_extractor_v4.md')
    extractor_path = Path(f'/root/提示词/专项提取器/{extractor_file}')
    extractor_prompt = extractor_path.read_text()
    extract_content, extract_time, extract_usage = call_qwen(
        extractor_prompt.replace("【文献内容】", f"【文献内容】\n{pdf_text}"), 4000)
    extract_json = parse_json(extract_content)
    print(f"  提取: {'成功' if extract_json else '失败'} ({extract_time:.1f}s)")
    result["extract"] = {"success": extract_json is not None, "result": extract_json, 
                         "time": extract_time, "tokens": extract_usage.get('total_tokens', 0)}
    
    # 3. GPT评估
    extract_str = json.dumps(extract_json, ensure_ascii=False, indent=2) if extract_json else "null"
    eval_prompt = f"""评估AI提取结果质量。

## 原文:
{pdf_text}

## 提取结果:
{extract_str}

## 评分(1-10):
1. 准确性 2. 完整性 3. 结构化 4. 页码准确 5. 无编造

输出JSON:
```json
{{"scores": {{"accuracy": 分数, "completeness": 分数, "structure": 分数, "page_accuracy": 分数, "no_hallucination": 分数, "overall": 平均}}, "summary": "总结"}}
```"""
    
    eval_content, eval_time, eval_usage = call_gpt52(eval_prompt)
    eval_json = parse_json(eval_content)
    score = eval_json.get('scores', {}).get('overall', 0) if eval_json else 0
    print(f"  评分: {score}/10 ({eval_time:.1f}s)")
    result["eval"] = {"score": score, "result": eval_json, "time": eval_time, "tokens": eval_usage.get('total_tokens', 0)}
    
    # 性能
    result["perf"] = {
        "qwen_time": classify_time + extract_time,
        "qwen_tokens": result["classify"]["tokens"] + result["extract"]["tokens"],
        "gpt_time": eval_time, "gpt_tokens": result["eval"]["tokens"]
    }
    
    # 保存
    with open(pdf_output_dir / "result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"batch10_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(f"批量测试 - 10个PDF")
    print(f"输出: {output_dir}")
    print("="*60)
    
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))[:NUM_TESTS]
    print(f"找到 {len(pdf_files)} 个PDF")
    
    start_time = time.time()
    results = []
    for i, pdf in enumerate(pdf_files, 1):
        results.append(process_pdf(pdf, output_dir, i, len(pdf_files)))
    
    total_time = time.time() - start_time
    
    # 汇总
    scores = [r['eval']['score'] for r in results if r['eval']['score']]
    types = {}
    for r in results:
        t = r['classify']['type']
        types[t] = types.get(t, 0) + 1
    
    qwen_times = [r['perf']['qwen_time'] for r in results]
    qwen_tokens = sum(r['perf']['qwen_tokens'] for r in results)
    gpt_tokens = sum(r['perf']['gpt_tokens'] for r in results)
    
    print("\n" + "="*60)
    print("汇总")
    print("="*60)
    print(f"总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
    print(f"类型分布: {types}")
    print(f"\n【Qwen性能】")
    print(f"  平均: {sum(qwen_times)/len(qwen_times):.1f}s/文件")
    print(f"  总tokens: {qwen_tokens}")
    print(f"\n【GPT评分】")
    print(f"  平均: {sum(scores)/len(scores):.1f}/10" if scores else "  无有效评分")
    print(f"  分布: {sorted(scores)}")
    print(f"  >=8分: {len([s for s in scores if s >= 8])}/{len(scores)}")
    print(f"  总tokens: {gpt_tokens}")
    
    summary = {
        "timestamp": timestamp, "total_files": len(results), "total_time": total_time,
        "type_distribution": types,
        "qwen": {"avg_time": sum(qwen_times)/len(qwen_times), "total_tokens": qwen_tokens},
        "gpt": {"avg_score": sum(scores)/len(scores) if scores else 0, "scores": scores, "total_tokens": gpt_tokens},
        "files": [{"name": r['filename'], "type": r['classify']['type'], "score": r['eval']['score']} for r in results]
    }
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果: {output_dir}")

if __name__ == '__main__':
    main()
