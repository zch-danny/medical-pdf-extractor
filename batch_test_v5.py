#!/usr/bin/env python3
"""
使用优化后的v5提取器进行测试
"""
import json
import time
import requests
import fitz
import shutil
from pathlib import Path
from datetime import datetime
import re

# 配置
QWEN_API_URL = "http://localhost:8000/v1/chat/completions"
GPT_API_URL = "https://api.bltcy.ai/v1/chat/completions"
GPT_API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
GPT_MODEL = "gpt-5.2"

PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUTPUT_ROOT = Path('/root/extraction_test_results')
NUM_TESTS = 5

# 使用新版提取器
EXTRACTOR_MAP = {
    'GUIDELINE': 'GUIDELINE_extractor_v5.md',  # 新版
    'META': 'META_extractor_v2.md',
    'REVIEW': 'REVIEW_extractor.md',
    'OTHER': 'OTHER_extractor_v3.md'  # 新版
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
        try:
            return json.loads(m.group())
        except:
            return None
    return None

def call_qwen(prompt, max_tokens=4000):
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    start = time.time()
    r = requests.post(QWEN_API_URL, json=payload, timeout=180)
    elapsed = time.time() - start
    result = r.json()
    content = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    return content, elapsed, usage

def call_gpt52(prompt, max_tokens=4000):
    headers = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GPT_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.3}
    start = time.time()
    r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start
    result = r.json()
    if "error" in result:
        return f"API错误: {result['error']}", elapsed, {}
    return result['choices'][0]['message']['content'], elapsed, result.get('usage', {})

def process_single_pdf(pdf_path, output_dir):
    print(f"\n{'='*50}\n处理: {pdf_path.name}\n{'='*50}")
    
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, pdf_output_dir / pdf_path.name)
    
    pdf_text = extract_pdf_text(pdf_path)
    result = {"filename": pdf_path.name, "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    # 1. 分类
    print("  [1/3] Qwen分类...")
    classifier_prompt = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
    full_prompt = classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}")
    classify_content, classify_time, classify_usage = call_qwen(full_prompt, 2000)
    classify_json = parse_json(classify_content)
    doc_type = classify_json.get('type', 'OTHER') if classify_json else 'OTHER'
    print(f"      类型: {doc_type}, 耗时: {classify_time:.1f}s")
    result["qwen_classify"] = {"result": classify_json, "time": classify_time, "tokens": classify_usage}
    
    # 2. 提取 (使用新版提取器)
    print(f"  [2/3] Qwen提取 (v5)...")
    extractor_file = EXTRACTOR_MAP.get(doc_type, 'OTHER_extractor_v3.md')
    extractor_path = Path(f'/root/提示词/专项提取器/{extractor_file}')
    
    if extractor_path.exists():
        extractor_prompt = extractor_path.read_text()
        full_prompt = extractor_prompt.replace("【文献内容】", f"【文献内容】\n{pdf_text}")
        extract_content, extract_time, extract_usage = call_qwen(full_prompt, 4000)
        extract_json = parse_json(extract_content)
        result["qwen_extract"] = {"result": extract_json, "raw": extract_content[:2000], "time": extract_time, "tokens": extract_usage}
        print(f"      耗时: {extract_time:.1f}s, JSON解析: {'成功' if extract_json else '失败'}")
    else:
        result["qwen_extract"] = {"error": f"提取器不存在: {extractor_file}"}
        extract_time, extract_usage = 0, {}
    
    # 3. GPT评估
    print("  [3/3] GPT-5.2评估...")
    eval_prompt = f"""评估以下AI提取结果的质量。

## 原始文献内容:
{pdf_text[:5000]}

## AI提取结果:
{json.dumps(result.get('qwen_extract', {}).get('result', {}), ensure_ascii=False, indent=2)[:2500]}

## 评估维度（1-10分）:
1. 准确性: 信息是否与原文一致
2. 完整性: 是否提取了关键信息
3. 结构化: JSON是否正确填充
4. 页码准确: sources页码是否正确
5. 无编造: 是否避免了虚构内容

输出JSON:
```json
{{"scores": {{"accuracy": 分数, "completeness": 分数, "structure": 分数, "page_accuracy": 分数, "no_hallucination": 分数, "overall": 平均分}}, "strengths": ["优点"], "weaknesses": ["问题"], "summary": "总结"}}
```"""
    
    eval_content, eval_time, eval_usage = call_gpt52(eval_prompt)
    eval_json = parse_json(eval_content)
    result["gpt52_evaluation"] = {"result": eval_json, "raw": eval_content, "time": eval_time, "tokens": eval_usage}
    score = eval_json.get('scores', {}).get('overall', 'N/A') if eval_json else 'N/A'
    print(f"      评分: {score}/10, 耗时: {eval_time:.1f}s")
    
    # 性能汇总
    result["performance"] = {
        "qwen_total_time": classify_time + result.get('qwen_extract', {}).get('time', 0),
        "qwen_total_tokens": classify_usage.get('total_tokens', 0) + extract_usage.get('total_tokens', 0),
        "gpt_time": eval_time, "gpt_tokens": eval_usage.get('total_tokens', 0)
    }
    
    # 保存
    with open(pdf_output_dir / "qwen_result.json", 'w', encoding='utf-8') as f:
        json.dump({"classify": result["qwen_classify"], "extract": result["qwen_extract"]}, f, ensure_ascii=False, indent=2)
    with open(pdf_output_dir / "gpt52_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(result["gpt52_evaluation"], f, ensure_ascii=False, indent=2)
    with open(pdf_output_dir / "full_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

def main():
    print("="*50 + "\n提取器v5测试 + GPT-5.2评估\n" + "="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"test_v5_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出: {output_dir}")
    
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))[:NUM_TESTS]
    all_results = []
    
    for i, pdf in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}]", end="")
        all_results.append(process_single_pdf(pdf, output_dir))
    
    # 汇总
    scores = [r['gpt52_evaluation']['result']['scores']['overall'] for r in all_results if r.get('gpt52_evaluation', {}).get('result', {}).get('scores')]
    qwen_times = [r['performance']['qwen_total_time'] for r in all_results]
    
    print("\n" + "="*50 + "\n汇总\n" + "="*50)
    print(f"测试数: {len(all_results)}")
    print(f"Qwen平均耗时: {sum(qwen_times)/len(qwen_times):.1f}s")
    print(f"GPT平均评分: {sum(scores)/len(scores):.1f}/10" if scores else "无有效评分")
    print(f"各文件评分: {scores}")
    
    summary = {"test_time": timestamp, "extractor_version": "v5", "results": [
        {"file": r['filename'], "type": r['qwen_classify']['result'].get('type') if r.get('qwen_classify',{}).get('result') else None,
         "score": r['gpt52_evaluation']['result']['scores']['overall'] if r.get('gpt52_evaluation',{}).get('result',{}).get('scores') else None}
        for r in all_results
    ], "avg_score": sum(scores)/len(scores) if scores else 0}
    
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存: {output_dir}")

if __name__ == '__main__':
    main()
