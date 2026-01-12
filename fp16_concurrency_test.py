#!/usr/bin/env python3
"""
测试不同并发数下的性能对比
"""
import json
import time
import requests
import fitz
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

API_URL = "http://localhost:8000/v1/chat/completions"
CLASSIFIER_PROMPT_PATH = Path('/root/提示词/分类器/classifier_prompt.md')
PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')

def extract_pdf_text(pdf_path, max_pages=5, max_chars=8000):
    doc = fitz.open(pdf_path)
    text_parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text_parts.append(f"\n[第{i+1}页]\n" + page.get_text())
    doc.close()
    full_text = ''.join(text_parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    return full_text

def parse_json_from_response(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    match = re.search(r'\{[\s\S]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            return None
    return None

def call_qwen_classify(pdf_text, timeout=60):
    classifier_prompt = CLASSIFIER_PROMPT_PATH.read_text(encoding='utf-8')
    full_prompt = classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{pdf_text}")
    
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, timeout=timeout)
    elapsed = time.time() - start_time
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    parsed = parse_json_from_response(content)
    
    return elapsed, parsed

def process_single_pdf_classify_only(pdf_path):
    """只测试分类阶段"""
    try:
        pdf_text = extract_pdf_text(pdf_path)
        elapsed, parsed = call_qwen_classify(pdf_text)
        return {
            'filename': pdf_path.name,
            'elapsed': elapsed,
            'type': parsed.get('type') if parsed else None,
            'success': True
        }
    except Exception as e:
        return {
            'filename': pdf_path.name,
            'error': str(e),
            'success': False
        }

def test_concurrency(concurrency, pdf_files):
    """测试指定并发数"""
    print(f"\n测试并发数: {concurrency}")
    print("-" * 40)
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process_single_pdf_classify_only, pdf): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result['success']:
                print(f"  ✓ {result['filename']}: {result['type']} ({result['elapsed']:.1f}s)")
    
    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]
    
    if successful:
        times = [r['elapsed'] for r in successful]
        avg_time = sum(times) / len(times)
        throughput = len(results) / (total_time / 60)
        
        print(f"\n并发{concurrency}结果:")
        print(f"  总耗时: {total_time:.1f}秒")
        print(f"  单文件平均: {avg_time:.1f}秒")
        print(f"  单文件范围: {min(times):.1f}s - {max(times):.1f}s")
        print(f"  吞吐量: {throughput:.2f} 文件/分钟")
        
        return {
            'concurrency': concurrency,
            'total_time': total_time,
            'avg_per_file': avg_time,
            'min_time': min(times),
            'max_time': max(times),
            'throughput': throughput,
            'success_rate': len(successful) / len(results)
        }
    return None

def main():
    print("=" * 60)
    print("fp16 并发性能对比测试 (仅分类阶段)")
    print("=" * 60)
    
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))
    print(f"测试文件数: {len(pdf_files)}")
    
    # 测试不同并发数
    concurrency_levels = [1, 2, 3, 5]
    all_results = []
    
    for conc in concurrency_levels:
        result = test_concurrency(conc, pdf_files)
        if result:
            all_results.append(result)
        time.sleep(2)  # 短暂休息
    
    # 汇总对比
    print("\n" + "=" * 60)
    print("并发性能对比汇总")
    print("=" * 60)
    print(f"{'并发数':<8} {'单文件速度':<12} {'吞吐量':<15} {'适用场景'}")
    print("-" * 60)
    
    for r in all_results:
        scenario = {
            1: "单用户最快",
            2: "用户上传推荐",
            3: "小批量处理",
            5: "批量处理最优"
        }.get(r['concurrency'], "")
        
        print(f"{r['concurrency']:<8} {r['avg_per_file']:.1f}s{'':<9} {r['throughput']:.2f} 文件/分钟{'':<4} {scenario}")
    
    # 保存结果
    output = {
        'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_files': len(pdf_files),
        'results': all_results
    }
    
    output_file = Path('fp16_concurrency_comparison.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
