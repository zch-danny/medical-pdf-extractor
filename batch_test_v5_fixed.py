#!/usr/bin/env python3
"""
修复版：评估时使用与提取相同的文本长度
"""
import json, time, requests, fitz, shutil, re
from pathlib import Path
from datetime import datetime

QWEN_API_URL = "http://localhost:8000/v1/chat/completions"
GPT_API_URL = "https://api.bltcy.ai/v1/chat/completions"
GPT_API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUTPUT_ROOT = Path('/root/extraction_test_results')
NUM_TESTS = 5

EXTRACTOR_MAP = {
    'GUIDELINE': 'GUIDELINE_extractor_v5.md',
    'META': 'META_extractor_v2.md',
    'REVIEW': 'REVIEW_extractor.md',
    'OTHER': 'OTHER_extractor_v3.md'
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
    r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=120)
    result = r.json()
    if "error" in result: return f"错误: {result['error']}", time.time() - start, {}
    return result['choices'][0]['message']['content'], time.time() - start, result.get('usage', {})

def process_pdf(pdf_path, output_dir):
    print(f"\n{'='*50}\n处理: {pdf_path.name}\n{'='*50}")
    
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, pdf_output_dir / pdf_path.name)
    
    # 提取文本 - 保存供评估使用
    pdf_text = extract_pdf_text(pdf_path)
    result = {"filename": pdf_path.name, "test_time": datetime.now().isoformat()}
    
    # 1. 分类
    print("  [1/3] Qwen分类...")
    classifier_prompt = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
    classify_content, classify_time, classify_usage = call_qwen(
        classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}"), 2000)
    classify_json = parse_json(classify_content)
    doc_type = classify_json.get('type', 'OTHER') if classify_json else 'OTHER'
    print(f"      类型: {doc_type}, 耗时: {classify_time:.1f}s")
    result["classify"] = {"result": classify_json, "time": classify_time, "tokens": classify_usage}
    
    # 2. 提取
    print(f"  [2/3] Qwen提取...")
    extractor_file = EXTRACTOR_MAP.get(doc_type, 'OTHER_extractor_v3.md')
    extractor_path = Path(f'/root/提示词/专项提取器/{extractor_file}')
    
    extractor_prompt = extractor_path.read_text()
    extract_content, extract_time, extract_usage = call_qwen(
        extractor_prompt.replace("【文献内容】", f"【文献内容】\n{pdf_text}"), 4000)
    extract_json = parse_json(extract_content)
    print(f"      耗时: {extract_time:.1f}s, 解析: {'成功' if extract_json else '失败'}")
    result["extract"] = {"result": extract_json, "time": extract_time, "tokens": extract_usage}
    
    # 3. GPT评估 - 修复：使用完整的pdf_text（与提取相同）
    print("  [3/3] GPT-5.2评估...")
    extract_json_str = json.dumps(extract_json, ensure_ascii=False, indent=2) if extract_json else "null"
    
    eval_prompt = f"""你是医学文献信息提取质量评估专家。

## 原始文献内容（与AI提取时使用的完全相同）:
{pdf_text}

## AI提取结果:
{extract_json_str}

## 评估任务
对比原文和提取结果，从以下5个维度评分（1-10分）：

1. **准确性**: 提取的信息是否与原文一致，有无错误
2. **完整性**: 是否提取了原文中的关键信息（标题、作者、推荐意见等）
3. **结构化**: JSON格式是否正确，字段是否合理填充
4. **页码准确**: sources中的页码是否与信息实际出现的页面匹配（对照[第X页]标记）
5. **无编造**: 是否严格基于原文，无虚构内容

## 输出要求
只输出JSON，格式如下：
```json
{{
  "scores": {{
    "accuracy": 分数,
    "completeness": 分数,
    "structure": 分数,
    "page_accuracy": 分数,
    "no_hallucination": 分数,
    "overall": 五项平均分
  }},
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "summary": "一句话总结"
}}
```"""
    
    eval_content, eval_time, eval_usage = call_gpt52(eval_prompt)
    eval_json = parse_json(eval_content)
    score = eval_json.get('scores', {}).get('overall', 'N/A') if eval_json else 'N/A'
    print(f"      评分: {score}/10, 耗时: {eval_time:.1f}s")
    result["evaluation"] = {"result": eval_json, "time": eval_time, "tokens": eval_usage}
    
    # 汇总
    result["performance"] = {
        "qwen_time": classify_time + extract_time,
        "qwen_tokens": classify_usage.get('total_tokens', 0) + extract_usage.get('total_tokens', 0),
        "gpt_time": eval_time,
        "gpt_tokens": eval_usage.get('total_tokens', 0)
    }
    
    # 保存
    for name, data in [("qwen_result", {"classify": result["classify"], "extract": result["extract"]}),
                       ("gpt52_evaluation", result["evaluation"]), ("full_result", result)]:
        with open(pdf_output_dir / f"{name}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    return result

def main():
    print("="*50 + "\n提取器v5测试（修复版评估）\n" + "="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"test_v5_fixed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))[:NUM_TESTS]
    results = [process_pdf(pdf, output_dir) for pdf in pdf_files]
    
    # 汇总
    scores = [r['evaluation']['result']['scores']['overall'] for r in results 
              if r.get('evaluation', {}).get('result', {}).get('scores')]
    
    print("\n" + "="*50 + "\n汇总\n" + "="*50)
    print(f"平均评分: {sum(scores)/len(scores):.1f}/10" if scores else "无评分")
    print(f"各文件: {scores}")
    
    summary = {"timestamp": timestamp, "avg_score": sum(scores)/len(scores) if scores else 0, "scores": scores}
    with open(output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"结果: {output_dir}")

if __name__ == '__main__':
    main()
