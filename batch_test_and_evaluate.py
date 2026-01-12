#!/usr/bin/env python3
"""
批量测试Qwen3-8B提取 + GPT-5.2评估
"""
import json
import time
import requests
import fitz
import shutil
from pathlib import Path
from datetime import datetime
import re

# ============ 配置 ============
QWEN_API_URL = "http://localhost:8000/v1/chat/completions"
GPT_API_URL = "https://api.bltcy.ai/v1/chat/completions"
GPT_API_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
GPT_MODEL = "gpt-5.2"

PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUTPUT_ROOT = Path('/root/extraction_test_results')
NUM_TESTS = 5  # 测试PDF数量

# ============ 工具函数 ============
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
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    start = time.time()
    r = requests.post(GPT_API_URL, headers=headers, json=payload, timeout=120)
    elapsed = time.time() - start
    result = r.json()
    
    if "error" in result:
        return f"API错误: {result['error']}", elapsed, {}
    
    content = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    return content, elapsed, usage

def process_single_pdf(pdf_path, output_dir):
    """处理单个PDF：提取 + 评估"""
    print(f"\n{'='*60}")
    print(f"处理: {pdf_path.name}")
    print('='*60)
    
    # 创建输出目录
    pdf_output_dir = output_dir / pdf_path.stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制PDF到输出目录
    shutil.copy2(pdf_path, pdf_output_dir / pdf_path.name)
    
    # 提取PDF文本
    pdf_text = extract_pdf_text(pdf_path)
    
    result = {
        "filename": pdf_path.name,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "qwen_classify": None,
        "qwen_extract": None,
        "gpt52_evaluation": None,
        "performance": {}
    }
    
    # ========== 1. Qwen分类 ==========
    print("  [1/3] Qwen分类中...")
    classifier_prompt = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
    full_prompt = classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}")
    
    classify_content, classify_time, classify_usage = call_qwen(full_prompt, 2000)
    classify_json = parse_json(classify_content)
    
    result["qwen_classify"] = {
        "result": classify_json,
        "time": classify_time,
        "tokens": classify_usage
    }
    print(f"      类型: {classify_json.get('type') if classify_json else 'FAILED'}")
    print(f"      耗时: {classify_time:.1f}s, tokens: {classify_usage.get('total_tokens', 0)}")
    
    # ========== 2. Qwen提取 ==========
    doc_type = classify_json.get('type', 'OTHER') if classify_json else 'OTHER'
    print(f"  [2/3] Qwen提取中 ({doc_type})...")
    
    extractor_map = {
        'GUIDELINE': 'GUIDELINE_extractor_v4.md',
        'META': 'META_extractor_v2.md',
        'REVIEW': 'REVIEW_extractor.md',
        'OTHER': 'OTHER_extractor_v2.md'
    }
    extractor_file = extractor_map.get(doc_type, 'OTHER_extractor_v2.md')
    extractor_path = Path(f'/root/提示词/专项提取器/{extractor_file}')
    
    if extractor_path.exists():
        extractor_prompt = extractor_path.read_text()
        if "【文献内容】" in extractor_prompt:
            full_prompt = extractor_prompt.replace("【文献内容】", f"【文献内容】\n{pdf_text}")
        else:
            full_prompt = extractor_prompt + f"\n\n【文献内容】\n{pdf_text}"
        
        extract_content, extract_time, extract_usage = call_qwen(full_prompt, 4000)
        extract_json = parse_json(extract_content)
        
        result["qwen_extract"] = {
            "result": extract_json,
            "time": extract_time,
            "tokens": extract_usage
        }
        print(f"      耗时: {extract_time:.1f}s, tokens: {extract_usage.get('total_tokens', 0)}")
    else:
        result["qwen_extract"] = {"error": f"提取器不存在: {extractor_file}"}
        extract_time = 0
        extract_usage = {}
    
    # ========== 3. GPT-5.2评估 ==========
    print("  [3/3] GPT-5.2评估中...")
    
    eval_prompt = f"""你是医学文献信息提取质量评估专家。请评估以下AI提取结果的质量。

## 原始PDF内容（前5页）:
{pdf_text[:6000]}

## AI提取结果:
{json.dumps(result.get('qwen_extract', {}).get('result', {}), ensure_ascii=False, indent=2)[:3000]}

## 评估任务
请从以下维度评估提取质量，每项打分1-10分：
1. **准确性**: 提取信息是否与原文一致
2. **完整性**: 是否提取了关键信息
3. **结构化**: JSON结构是否合理
4. **推荐意见**: 推荐意见提取是否准确
5. **元数据**: 标题、机构等是否正确

## 输出JSON格式:
```json
{{
  "scores": {{"accuracy": 分数, "completeness": 分数, "structure": 分数, "recommendations": 分数, "metadata": 分数, "overall": 平均分}},
  "strengths": ["优点"],
  "weaknesses": ["问题"],
  "missing_info": ["遗漏"],
  "summary": "一句话总结"
}}
```
"""
    
    eval_content, eval_time, eval_usage = call_gpt52(eval_prompt)
    eval_json = parse_json(eval_content)
    
    result["gpt52_evaluation"] = {
        "result": eval_json,
        "raw": eval_content,
        "time": eval_time,
        "tokens": eval_usage
    }
    
    overall_score = eval_json.get('scores', {}).get('overall', 'N/A') if eval_json else 'N/A'
    print(f"      评分: {overall_score}/10")
    print(f"      耗时: {eval_time:.1f}s, tokens: {eval_usage.get('total_tokens', 0)}")
    
    # ========== 汇总性能 ==========
    result["performance"] = {
        "qwen_classify_time": classify_time,
        "qwen_extract_time": result.get('qwen_extract', {}).get('time', 0),
        "qwen_total_time": classify_time + result.get('qwen_extract', {}).get('time', 0),
        "qwen_total_tokens": classify_usage.get('total_tokens', 0) + extract_usage.get('total_tokens', 0),
        "gpt52_eval_time": eval_time,
        "gpt52_eval_tokens": eval_usage.get('total_tokens', 0)
    }
    
    # ========== 保存结果 ==========
    # 保存Qwen结果
    qwen_result_file = pdf_output_dir / "qwen_result.json"
    with open(qwen_result_file, 'w', encoding='utf-8') as f:
        json.dump({
            "classify": result["qwen_classify"],
            "extract": result["qwen_extract"]
        }, f, ensure_ascii=False, indent=2)
    
    # 保存GPT评估结果
    eval_result_file = pdf_output_dir / "gpt52_evaluation.json"
    with open(eval_result_file, 'w', encoding='utf-8') as f:
        json.dump(result["gpt52_evaluation"], f, ensure_ascii=False, indent=2)
    
    # 保存完整结果
    full_result_file = pdf_output_dir / "full_result.json"
    with open(full_result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 结果已保存到: {pdf_output_dir}")
    
    return result

def main():
    print("=" * 60)
    print("Qwen3-8B 提取 + GPT-5.2 评估 批量测试")
    print("=" * 60)
    
    # 创建输出根目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 获取PDF文件
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))[:NUM_TESTS]
    print(f"测试文件数: {len(pdf_files)}")
    
    # 批量处理
    all_results = []
    overall_start = time.time()
    
    for i, pdf in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}]", end="")
        result = process_single_pdf(pdf, output_dir)
        all_results.append(result)
    
    overall_time = time.time() - overall_start
    
    # ========== 生成汇总报告 ==========
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    # 统计
    qwen_times = [r['performance']['qwen_total_time'] for r in all_results]
    qwen_tokens = [r['performance']['qwen_total_tokens'] for r in all_results]
    gpt_times = [r['performance']['gpt52_eval_time'] for r in all_results]
    gpt_tokens = [r['performance']['gpt52_eval_tokens'] for r in all_results]
    scores = [r['gpt52_evaluation']['result']['scores']['overall'] 
              for r in all_results 
              if r.get('gpt52_evaluation', {}).get('result', {}).get('scores')]
    
    summary = {
        "test_time": timestamp,
        "total_pdfs": len(all_results),
        "total_time": overall_time,
        "qwen_performance": {
            "avg_time": sum(qwen_times) / len(qwen_times),
            "min_time": min(qwen_times),
            "max_time": max(qwen_times),
            "avg_tokens": sum(qwen_tokens) // len(qwen_tokens),
            "total_tokens": sum(qwen_tokens)
        },
        "gpt52_evaluation": {
            "avg_time": sum(gpt_times) / len(gpt_times),
            "avg_tokens": sum(gpt_tokens) // len(gpt_tokens),
            "total_tokens": sum(gpt_tokens),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "scores": scores
        },
        "results": [{
            "filename": r['filename'],
            "type": r['qwen_classify']['result'].get('type') if r.get('qwen_classify', {}).get('result') else None,
            "qwen_time": r['performance']['qwen_total_time'],
            "qwen_tokens": r['performance']['qwen_total_tokens'],
            "gpt_score": r['gpt52_evaluation']['result']['scores']['overall'] if r.get('gpt52_evaluation', {}).get('result', {}).get('scores') else None
        } for r in all_results]
    }
    
    print(f"总测试数: {summary['total_pdfs']}")
    print(f"总耗时: {summary['total_time']:.1f}秒")
    print(f"\n【Qwen3-8B 性能】")
    print(f"  平均耗时: {summary['qwen_performance']['avg_time']:.1f}秒/文件")
    print(f"  平均tokens: {summary['qwen_performance']['avg_tokens']}")
    print(f"  总tokens: {summary['qwen_performance']['total_tokens']}")
    print(f"\n【GPT-5.2 评估】")
    print(f"  平均耗时: {summary['gpt52_evaluation']['avg_time']:.1f}秒/文件")
    print(f"  平均tokens: {summary['gpt52_evaluation']['avg_tokens']}")
    print(f"  平均评分: {summary['gpt52_evaluation']['avg_score']:.1f}/10")
    print(f"  各文件评分: {summary['gpt52_evaluation']['scores']}")
    
    # 保存汇总
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总报告已保存到: {summary_file}")
    print(f"完整结果目录: {output_dir}")

if __name__ == '__main__':
    main()
