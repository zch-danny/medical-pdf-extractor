#!/usr/bin/env python3
import json, time, requests, fitz, re
from pathlib import Path

API_URL = "http://localhost:8000/v1/chat/completions"
PDF_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')

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
    return json.loads(m.group()) if m else None

def call_api(prompt, max_tokens=4000):
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    start = time.time()
    r = requests.post(API_URL, json=payload, timeout=180)
    return r.json()['choices'][0]['message']['content'], time.time() - start

# 取第一个PDF测试
pdf = list(PDF_DIR.glob('*.pdf'))[0]
print(f"测试文件: {pdf.name}\n")

pdf_text = extract_pdf_text(pdf)

# 1. 分类
print("=== 分类阶段 ===")
classifier_prompt = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
full_prompt = classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}")
classify_result, classify_time = call_api(full_prompt, 2000)
classify_json = parse_json(classify_result)
print(f"耗时: {classify_time:.1f}s")
print(f"分类结果:\n{json.dumps(classify_json, ensure_ascii=False, indent=2)}\n")

# 2. 提取
doc_type = classify_json.get('type', 'OTHER')
print(f"=== 提取阶段 ({doc_type}) ===")
extractor_map = {
    'GUIDELINE': 'GUIDELINE_extractor_v4.md',
    'META': 'META_extractor_v2.md',
    'REVIEW': 'REVIEW_extractor.md',
    'OTHER': 'OTHER_extractor_v2.md'
}
extractor_file = extractor_map.get(doc_type, 'OTHER_extractor_v2.md')
extractor_prompt = Path(f'/root/提示词/专项提取器/{extractor_file}').read_text()
full_prompt = extractor_prompt.replace("【文献全文】", f"【文献全文】\n{pdf_text}")
extract_result, extract_time = call_api(full_prompt, 4000)
extract_json = parse_json(extract_result)
print(f"耗时: {extract_time:.1f}s")
print(f"提取结果:\n{json.dumps(extract_json, ensure_ascii=False, indent=2)}")

# 保存完整结果
output = {
    "filename": pdf.name,
    "classify": {"result": classify_json, "time": classify_time},
    "extract": {"result": extract_json, "time": extract_time, "raw": extract_result}
}
with open('extract_sample_output.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n完整结果已保存到: extract_sample_output.json")
