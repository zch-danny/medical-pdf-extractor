#!/usr/bin/env python3
"""
fp16 性能测试脚本
测试分类和提取两个阶段的速度和准确度
"""
import json
import time
import requests
import fitz
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# 配置
API_URL = "http://localhost:8000/v1/chat/completions"
CLASSIFIER_PROMPT_PATH = Path('/root/提示词/分类器/classifier_prompt.md')
PDF_SAMPLES_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
RESULTS_FILE = Path('fp16_test_results.json')

# 并发配置
CONCURRENCY = 2  # 根据memory.md推荐配置

def load_prompt(prompt_path):
    """加载提示词"""
    return prompt_path.read_text(encoding='utf-8')

def extract_pdf_text(pdf_path, max_pages=5, max_chars=8000):
    """提取PDF文本"""
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
    """从响应中解析JSON"""
    # 移除markdown代码块标记
    text = re.sub(r'```json\s*|\s*```', '', text)
    # 提取JSON对象
    match = re.search(r'\{[\s\S]+\}', text)
    if match:
        return json.loads(match.group())
    return None

def call_qwen_classify(pdf_text, timeout=60):
    """调用Qwen进行分类"""
    classifier_prompt = load_prompt(CLASSIFIER_PROMPT_PATH)
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
    usage = result.get('usage', {})
    
    parsed = parse_json_from_response(content)
    
    return {
        'parsed': parsed,
        'elapsed': elapsed,
        'usage': usage,
        'raw_content': content
    }

def call_qwen_extract(pdf_text, doc_type, timeout=180):
    """调用Qwen进行信息提取"""
    # 根据类型选择提取器
    extractor_map = {
        'GUIDELINE': 'GUIDELINE_extractor_v4.md',
        'META': 'META_extractor_v2.md',
        'REVIEW': 'REVIEW_extractor.md',
        'RCT': 'RCT_extractor_v2.md',
        'OTHER': 'OTHER_extractor_v2.md'
    }
    
    extractor_file = extractor_map.get(doc_type, 'OTHER_extractor_v2.md')
    extractor_path = Path(f'/root/提示词/专项提取器/{extractor_file}')
    
    if not extractor_path.exists():
        return {'error': f'提取器文件不存在: {extractor_path}'}
    
    extractor_prompt = extractor_path.read_text(encoding='utf-8')
    full_prompt = extractor_prompt.replace("【文献全文】", f"【文献全文】\n{pdf_text}")
    
    payload = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 4000,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=payload, timeout=timeout)
    elapsed = time.time() - start_time
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    usage = result.get('usage', {})
    
    parsed = parse_json_from_response(content)
    
    return {
        'parsed': parsed,
        'elapsed': elapsed,
        'usage': usage,
        'raw_content': content[:500]  # 只保存前500字符
    }

def process_single_pdf(pdf_path):
    """处理单个PDF的完整流程"""
    print(f"\n处理: {pdf_path.name}")
    
    result = {
        'filename': pdf_path.name,
        'classify': None,
        'extract': None,
        'total_time': 0,
        'success': False
    }
    
    try:
        # 提取PDF文本
        pdf_text = extract_pdf_text(pdf_path)
        
        # 阶段1: 分类
        print(f"  [1/2] 分类中...")
        classify_result = call_qwen_classify(pdf_text)
        result['classify'] = {
            'elapsed': classify_result['elapsed'],
            'type': classify_result['parsed'].get('type') if classify_result['parsed'] else None,
            'tokens': classify_result['usage'].get('total_tokens', 0)
        }
        print(f"  分类完成: {result['classify']['type']} ({classify_result['elapsed']:.1f}s, {result['classify']['tokens']} tokens)")
        
        # 阶段2: 提取
        if classify_result['parsed'] and classify_result['parsed'].get('type'):
            doc_type = classify_result['parsed']['type']
            print(f"  [2/2] 提取中 ({doc_type})...")
            extract_result = call_qwen_extract(pdf_text, doc_type)
            
            if 'error' not in extract_result:
                result['extract'] = {
                    'elapsed': extract_result['elapsed'],
                    'success': extract_result['parsed'] is not None,
                    'tokens': extract_result['usage'].get('total_tokens', 0)
                }
                print(f"  提取完成: {'成功' if result['extract']['success'] else '失败'} ({extract_result['elapsed']:.1f}s, {result['extract']['tokens']} tokens)")
            else:
                result['extract'] = {'error': extract_result['error']}
                print(f"  提取错误: {extract_result['error']}")
        
        result['total_time'] = (result['classify']['elapsed'] + 
                               (result['extract']['elapsed'] if result['extract'] and 'elapsed' in result['extract'] else 0))
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  ❌ 错误: {e}")
    
    return result

def main():
    print("=" * 60)
    print("fp16 性能测试 - 完整两阶段流程")
    print("=" * 60)
    print(f"并发数: {CONCURRENCY}")
    print(f"输入配置: 5页, 8000字符")
    print(f"测试目录: {PDF_SAMPLES_DIR}")
    print()
    
    # 获取所有PDF文件
    pdf_files = list(PDF_SAMPLES_DIR.glob('*.pdf'))
    print(f"找到 {len(pdf_files)} 个PDF文件")
    
    if not pdf_files:
        print("❌ 未找到PDF文件!")
        return
    
    # 执行测试
    overall_start = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    overall_elapsed = time.time() - overall_start
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    print(f"总文件数: {len(results)}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(results) - len(successful)}")
    print(f"总耗时: {overall_elapsed:.1f}秒")
    print(f"吞吐量: {len(results) / (overall_elapsed / 60):.2f} 文件/分钟")
    
    if successful:
        # 分类阶段统计
        classify_times = [r['classify']['elapsed'] for r in successful if r['classify']]
        classify_tokens = [r['classify']['tokens'] for r in successful if r['classify']]
        
        print(f"\n【分类阶段】")
        print(f"  平均耗时: {sum(classify_times)/len(classify_times):.1f}秒")
        print(f"  范围: {min(classify_times):.1f}s - {max(classify_times):.1f}s")
        print(f"  平均tokens: {sum(classify_tokens)//len(classify_tokens)}")
        
        # 提取阶段统计
        extract_results = [r['extract'] for r in successful if r.get('extract') and 'elapsed' in r['extract']]
        if extract_results:
            extract_times = [e['elapsed'] for e in extract_results]
            extract_tokens = [e['tokens'] for e in extract_results]
            extract_success = [e for e in extract_results if e['success']]
            
            print(f"\n【提取阶段】")
            print(f"  成功率: {len(extract_success)}/{len(extract_results)} ({100*len(extract_success)/len(extract_results):.1f}%)")
            print(f"  平均耗时: {sum(extract_times)/len(extract_times):.1f}秒")
            print(f"  范围: {min(extract_times):.1f}s - {max(extract_times):.1f}s")
            print(f"  平均tokens: {sum(extract_tokens)//len(extract_tokens)}")
        
        # 完整流程统计
        total_times = [r['total_time'] for r in successful]
        print(f"\n【完整流程】")
        print(f"  平均耗时: {sum(total_times)/len(total_times):.1f}秒/文件")
        print(f"  范围: {min(total_times):.1f}s - {max(total_times):.1f}s")
        
        # 类型分布
        types = [r['classify']['type'] for r in successful if r['classify'] and r['classify']['type']]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\n【类型分布】")
        for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count}")
    
    # 保存结果
    output = {
        'test_config': {
            'concurrency': CONCURRENCY,
            'input_pages': 5,
            'input_chars': 8000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'summary': {
            'total_files': len(results),
            'successful': len(successful),
            'total_time': overall_elapsed,
            'throughput': len(results) / (overall_elapsed / 60)
        },
        'details': results
    }
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {RESULTS_FILE}")

if __name__ == '__main__':
    main()
