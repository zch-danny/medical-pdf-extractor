#!/usr/bin/env python3
"""
医学摘要生成质量评估脚本
使用GPT-5.2对摘要进行准确度、完整性、可读性评估
"""
import json
import time
import requests
import fitz
import importlib.util
from pathlib import Path
from typing import Dict, List
import sys
import re

# API配置
API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
API_DOMAIN = "api.bltcy.ai"
MODEL = "gpt-5.2"

def call_gpt52(prompt: str, max_tokens: int = 2000) -> str:
    """调用GPT-5.2 API"""
    url = f"https://{API_DOMAIN}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    return response.json()['choices'][0]['message']['content']

def smart_extract_pdf_text(pdf_path: str, max_chars: int = 30000) -> str:
    """智能提取PDF文本，跳过目录，提取正文"""
    doc = fitz.open(pdf_path)
    full_text = '\n'.join(page.get_text() for page in doc)
    doc.close()
    
    total_len = len(full_text)
    if total_len <= max_chars:
        return full_text
    
    # 尝试找到正文开始位置
    content_start = 0
    patterns = [
        r'1\.\s*Preamble\s+(?:Guidelines|This|The)',
        r'1\.\s*Introduction\s+(?:The|This|In)',
        r'Preamble\s+(?:Guidelines|This|The)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            content_start = match.start()
            break
    
    # 如果找到正文开始位置且与开头差距较大
    if content_start > 5000:
        meta_text = full_text[:5000]
        content_text = full_text[content_start:content_start + max_chars - 5000]
        return meta_text + "\n...[目录省略]...\n" + content_text
    
    return full_text[:max_chars]

def evaluate_summary(original_text: str, summary_result: Dict) -> Dict:
    """使用GPT-5.2评估摘要质量"""
    summary = summary_result.get('summary', '')
    key_points = summary_result.get('key_points', [])
    
    eval_prompt = f"""你是医学文献摘要质量评估专家。请严格评估以下AI生成摘要的质量。

## 原始PDF内容（部分）:
{original_text}

## AI生成的摘要:
{summary}

## AI提取的关键要点:
{json.dumps(key_points, ensure_ascii=False, indent=2)}

## 评估维度（每项1-10分）:

1. **准确度 (Accuracy)**: 摘要是否与原文一致？是否有臆测、扩写或添加原文没有的内容？
   - 10分: 完全忠于原文，所有信息都能在原文找到
   - 7分: 基本准确，偶有小的表述偏差
   - 5分: 部分准确，有明显添加或错误
   - 3分: 错误较多或有严重臆测
   - 1分: 大量编造或与原文矛盾

2. **完整性 (Completeness)**: 是否覆盖了原文的核心信息？
   - 10分: 完整覆盖研究背景、目的、方法、结果、结论
   - 7分: 覆盖了主要信息，遗漏少量次要内容
   - 5分: 覆盖部分核心内容
   - 3分: 遗漏重要信息
   - 1分: 信息严重缺失

3. **可读性 (Readability)**: 语言是否流畅、结构是否清晰？
   - 10分: 专业流畅，结构清晰，易于理解
   - 7分: 语言通顺，结构基本合理
   - 5分: 可以理解但有改进空间
   - 3分: 较难理解或结构混乱
   - 1分: 语言混乱

## 输出JSON格式（必须严格遵守）:
```json
{{
  "accuracy": 数字,
  "completeness": 数字,
  "readability": 数字,
  "overall": 数字(三项平均),
  "accuracy_issues": ["准确度问题1", "问题2"],
  "missing_info": ["遗漏信息1", "遗漏2"],
  "hallucinations": ["臆测/编造内容1"],
  "brief_comment": "一句话评价"
}}
```"""
    
    response = call_gpt52(eval_prompt)
    
    m = re.search(r'\{[\s\S]*\}', response)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {"accuracy": 5, "completeness": 5, "readability": 5, "overall": 5, "brief_comment": "解析失败"}

def load_summarizer(module_path: str):
    """动态加载摘要器模块"""
    spec = importlib.util.spec_from_file_location("summarizer_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MedicalSummarizer()

def run_evaluation(summarizer_path: str):
    """运行评估"""
    module_file = Path(summarizer_path)
    if not module_file.exists():
        module_file = Path(f"/root/pdf_summarization_deploy_20251225_093847/{summarizer_path}")
    
    summarizer = load_summarizer(str(module_file))
    
    pdf_dir = Path('/root/autodl-tmp/pdf_input/pdf_input_09-1125')
    if not pdf_dir.exists():
        pdf_dir = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
    
    test_pdfs = sorted(pdf_dir.glob('*.pdf'))[:3]
    
    results = []
    print(f"\n{'='*60}")
    print(f"摘要评估 - {summarizer.VERSION}")
    print(f"{'='*60}\n")
    
    for pdf_path in test_pdfs:
        print(f"处理: {pdf_path.name}")
        
        t0 = time.time()
        summary_result = summarizer.generate_summary(str(pdf_path))
        gen_time = time.time() - t0
        
        if not summary_result.get('success') or not summary_result.get('summary'):
            print(f"  [跳过] 摘要生成失败")
            continue
        
        # 使用智能提取，与摘要器对齐
        original_text = smart_extract_pdf_text(str(pdf_path))
        
        print(f"  摘要字数: {len(summary_result['summary'])}, 原文提取: {len(original_text)}字符")
        print(f"  调用GPT-5.2评估...")
        
        t1 = time.time()
        eval_result = evaluate_summary(original_text, summary_result)
        eval_time = time.time() - t1
        
        result = {
            "pdf": pdf_path.name,
            "summary_length": len(summary_result['summary']),
            "facts_count": summary_result.get('facts_count', 0),
            "mode": summary_result.get('mode', ''),
            "gen_time": gen_time,
            "eval_time": eval_time,
            "scores": {
                "accuracy": eval_result.get('accuracy', 0),
                "completeness": eval_result.get('completeness', 0),
                "readability": eval_result.get('readability', 0),
                "overall": eval_result.get('overall', 0)
            },
            "issues": {
                "accuracy_issues": eval_result.get('accuracy_issues', []),
                "missing_info": eval_result.get('missing_info', []),
                "hallucinations": eval_result.get('hallucinations', [])
            },
            "comment": eval_result.get('brief_comment', '')
        }
        results.append(result)
        
        print(f"  评分: 准确={result['scores']['accuracy']:.1f}, 完整={result['scores']['completeness']:.1f}, 可读={result['scores']['readability']:.1f}, 总分={result['scores']['overall']:.1f}")
        print(f"  评价: {result['comment']}")
        print()
    
    if results:
        avg_accuracy = sum(r['scores']['accuracy'] for r in results) / len(results)
        avg_completeness = sum(r['scores']['completeness'] for r in results) / len(results)
        avg_readability = sum(r['scores']['readability'] for r in results) / len(results)
        avg_overall = sum(r['scores']['overall'] for r in results) / len(results)
        
        print(f"{'='*60}")
        print(f"汇总 ({len(results)}个PDF)")
        print(f"{'='*60}")
        print(f"平均准确度: {avg_accuracy:.2f}")
        print(f"平均完整性: {avg_completeness:.2f}")
        print(f"平均可读性: {avg_readability:.2f}")
        print(f"平均总分: {avg_overall:.2f}")
        
        output = {
            "version": summarizer.VERSION,
            "pdf_count": len(results),
            "averages": {
                "accuracy": avg_accuracy,
                "completeness": avg_completeness,
                "readability": avg_readability,
                "overall": avg_overall
            },
            "details": results
        }
        output_file = f"summarizer_{Path(summarizer_path).stem}_eval.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "medical_summarizer_v7.py"
    run_evaluation(version)
