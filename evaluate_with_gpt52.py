#!/usr/bin/env python3
"""
使用GPT-5.2评估Qwen提取结果的质量
"""
import json
import requests
import fitz
from pathlib import Path

# API配置 - 使用同一个代理
API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
API_DOMAIN = "api.bltcy.ai"
MODEL = "gpt-5.2"

def call_gpt52(prompt, max_tokens=4000):
    """调用GPT-5.2 API"""
    url = f"https://{API_DOMAIN}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    result = response.json()
    
    if "error" in result:
        return f"API错误: {result['error']}"
    
    return result['choices'][0]['message']['content']

def extract_pdf_text(pdf_path, max_pages=5, max_chars=8000):
    """提取PDF文本"""
    doc = fitz.open(pdf_path)
    text_parts = []
    for i, page in enumerate(doc):
        if i >= max_pages: break
        text_parts.append(f"\n[第{i+1}页]\n" + page.get_text())
    doc.close()
    return ''.join(text_parts)[:max_chars]

def main():
    # 加载Qwen提取结果
    qwen_result_file = Path('extract_sample_output.json')
    if not qwen_result_file.exists():
        print("错误: 找不到Qwen提取结果文件")
        return
    
    with open(qwen_result_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    
    # 获取原始PDF文本
    pdf_path = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples') / qwen_data['filename']
    original_text = extract_pdf_text(pdf_path)
    
    print(f"评估文件: {qwen_data['filename']}")
    print(f"Qwen分类: {qwen_data['classify'].get('type')}")
    print(f"调用GPT-5.2进行评估...\n")
    
    # 构建评估提示词
    eval_prompt = f"""你是医学文献信息提取质量评估专家。请评估以下AI提取结果的质量。

## 原始PDF内容（前5页）:
{original_text}

## AI提取结果:
{json.dumps(qwen_data['extract'], ensure_ascii=False, indent=2)}

## 评估任务
请从以下维度评估提取质量，每项打分1-10分：

1. **准确性** (Accuracy): 提取的信息是否与原文一致，有无错误
2. **完整性** (Completeness): 是否提取了原文中的关键信息
3. **结构化程度** (Structure): JSON结构是否合理，字段是否正确填充
4. **推荐意见提取** (Recommendations): 推荐意见是否完整、准确提取
5. **元数据提取** (Metadata): 标题、作者、机构等元数据是否正确

## 输出格式
请输出JSON格式的评估结果：
```json
{{
  "scores": {{
    "accuracy": 分数,
    "completeness": 分数,
    "structure": 分数,
    "recommendations": 分数,
    "metadata": 分数,
    "overall": 总分(平均)
  }},
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["问题1", "问题2"],
  "missing_info": ["遗漏信息1", "遗漏信息2"],
  "errors": ["错误1", "错误2"],
  "suggestions": ["改进建议1", "改进建议2"],
  "summary": "一句话总结评估结果"
}}
```
"""

    # 调用GPT-5.2评估
    eval_result = call_gpt52(eval_prompt)
    
    print("=" * 60)
    print("GPT-5.2 评估结果")
    print("=" * 60)
    print(eval_result)
    
    # 保存评估结果
    output = {
        "filename": qwen_data['filename'],
        "qwen_classify": qwen_data['classify'],
        "qwen_extract": qwen_data['extract'],
        "gpt52_evaluation": eval_result,
        "model_used": MODEL
    }
    
    output_file = Path('gpt52_evaluation_result.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n评估结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
