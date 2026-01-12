#!/usr/bin/env python3
"""
使用GPT-5.2生成高质量few-shot标注样本
为每种文献类型生成1-2个标准提取示例
"""
import json, requests, fitz, re
from pathlib import Path
from datetime import datetime

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

PDF_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUTPUT_DIR = Path('/root/pdf_summarization_deploy_20251225_093847/fewshot_samples')

def get_text(path, pages=5, chars=8000):
    doc = fitz.open(path)
    parts = [f"\n[第{i+1}页]\n" + doc[i].get_text() for i in range(min(pages, len(doc)))]
    doc.close()
    return ''.join(parts)[:chars]

def call_gpt(prompt, max_tokens=4000):
    """调用GPT-5.2生成标注"""
    r = requests.post(GPT_API, 
        headers={"Authorization": f"Bearer {GPT_KEY}"},
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], 
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=180)
    d = r.json()
    if "error" in d:
        print(f"API错误: {d['error']}")
        return None
    return d['choices'][0]['message']['content']

def parse_json(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    m = re.search(r'\{[\s\S]+\}', text)
    try: return json.loads(m.group()) if m else None
    except: return None

ANNOTATION_PROMPT = """你是医学文献信息提取专家。请为以下文献生成高质量的结构化提取结果，作为few-shot示例。

## 原始文献内容
{text}

## 文献类型
{doc_type}

## 任务
生成一个**完美的**提取示例，要求：
1. **页码100%准确** - 每条信息的sources必须与原文[第X页]标记完全对应
2. **不编造任何信息** - 只提取原文明确存在的内容
3. **完整提取关键信息** - 包括所有推荐意见、关键数据
4. **格式规范** - JSON结构清晰

## 输出格式
```json
{{
  "doc_metadata": {{
    "title": "精确标题",
    "authors": "作者或Not Found",
    "organization": "发布机构",
    "publish_date": "发布日期",
    "year": "年份",
    "doi": "DOI或Not Found",
    "document_type": "精确类型",
    "sources": ["实际页码如p1"]
  }},
  "scope": {{
    "objective": "文献目的",
    "target_population": "目标人群",
    "sources": ["页码"]
  }},
  "recommendations": [
    {{
      "id": "编号",
      "text": "原文逐字复制的推荐内容",
      "sources": ["该推荐实际出现的页码"]
    }}
  ],
  "key_findings": [
    {{
      "finding": "关键发现",
      "data": "具体数据(如有)",
      "sources": ["页码"]
    }}
  ],
  "extraction_notes": {{
    "completeness": "High/Medium/Low",
    "pages_with_key_content": ["列出包含关键内容的页码"]
  }}
}}
```

请生成这个完美的提取示例，只输出JSON。"""

# 为每种类型选择代表性PDF
TYPE_PDF_MAP = {
    "GUIDELINE": "1756864558155_1608702.pdf",  # NICE指南
    "OTHER": "1757066770930_1608702.pdf",       # WHO文档
    "REVIEW": "1760091187079_1608702.pdf",      # 综述
}

def generate_sample_for_type(doc_type: str, pdf_name: str):
    """为特定类型生成few-shot样本"""
    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}")
        return None
    
    print(f"\n生成 {doc_type} 样本: {pdf_name}")
    text = get_text(pdf_path)
    
    prompt = ANNOTATION_PROMPT.replace("{text}", text).replace("{doc_type}", doc_type)
    
    print("  调用GPT-5.2...")
    response = call_gpt(prompt)
    if not response:
        return None
    
    result = parse_json(response)
    if not result:
        print("  JSON解析失败")
        return None
    
    print("  生成成功")
    return {
        "type": doc_type,
        "pdf": pdf_name,
        "input_text": text,
        "expected_output": result
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GPT-5.2 生成高质量Few-shot样本")
    print("="*60)
    
    samples = {}
    
    for doc_type, pdf_name in TYPE_PDF_MAP.items():
        sample = generate_sample_for_type(doc_type, pdf_name)
        if sample:
            samples[doc_type] = sample
            
            # 单独保存每个类型的样本
            with open(OUTPUT_DIR / f"{doc_type}_sample.json", 'w') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            print(f"  已保存: {OUTPUT_DIR}/{doc_type}_sample.json")
    
    # 保存汇总
    with open(OUTPUT_DIR / "all_samples.json", 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 共生成 {len(samples)} 个样本")
    print(f"保存位置: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
