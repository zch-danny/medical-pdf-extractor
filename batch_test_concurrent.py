#!/usr/bin/env python3
"""并发测试v7.9 - 提取串行(共享vLLM)，评分并发"""

import sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v79 import MedicalPDFExtractor
from stable_evaluator import evaluate_stable

TEST_PDFS = [
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711187745_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711584225_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453891288_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694891048_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694954932_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756710561227_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711019517_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756710006526_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756705897161_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756709847051_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460577061_8106296.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756706418200_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756706468788_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460835542_8106296.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756710859463_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453814456_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453929400_8248372.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711869723_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711687858_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711943364_1608702.pdf",
]

def extract_one(extractor, pdf_path, idx):
    """提取单个PDF"""
    try:
        t0 = time.time()
        result = extractor.extract(pdf_path)
        return idx, pdf_path, result, time.time() - t0
    except Exception as e:
        return idx, pdf_path, {'success': False, 'error': str(e)}, 0

def evaluate_one(pdf_path, data):
    """评分单个结果"""
    try:
        return evaluate_stable(pdf_path, data, n_runs=1)
    except:
        return {'overall': 0}

def main():
    print(f"{'='*60}")
    print(f"并发测试 v7.9 - 20个PDF")
    print(f"{'='*60}")
    
    extractor = MedicalPDFExtractor()
    extractions = []
    
    # 阶段1: 串行提取 (vLLM单GPU)
    print("\n[阶段1] 串行提取...")
    t_extract = time.time()
    for i, pdf in enumerate(TEST_PDFS):
        print(f"  [{i+1:2d}/20] 提取中...", end=" ", flush=True)
        idx, path, result, dur = extract_one(extractor, pdf, i)
        success = result.get('success', False)
        data = result.get('data', result.get('result', {}))
        items = len(data.get('recommendations', [])) + len(data.get('key_findings', [])) if success else 0
        print(f"{'✓' if success else '✗'} {items}条 {dur:.1f}s")
        extractions.append((pdf, result, dur))
    print(f"  提取完成: {time.time()-t_extract:.1f}s")
    
    # 阶段2: 并发GPT评分
    print("\n[阶段2] 并发GPT评分 (4线程)...")
    t_eval = time.time()
    results = []
    
    eval_tasks = []
    for pdf, extraction, dur in extractions:
        if extraction.get('success'):
            data = extraction.get('data', extraction.get('result', {}))
            eval_tasks.append((pdf, data, dur))
        else:
            results.append({
                'pdf': pdf.split('/')[-1], 'success': False,
                'error': extraction.get('error')
            })
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_one, pdf, data): (pdf, data, dur) 
                   for pdf, data, dur in eval_tasks}
        
        done = 0
        for future in as_completed(futures):
            pdf, data, dur = futures[future]
            done += 1
            scores = future.result()
            items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
            
            if scores.get('overall', 0) > 0:
                print(f"  [{done:2d}] {pdf.split('/')[-1][:15]}... overall={scores['overall']}")
                results.append({
                    'pdf': pdf.split('/')[-1], 'success': True, 'items': items,
                    'extract_time': dur, 'scores': {
                        'accuracy': scores['accuracy'],
                        'completeness': scores['completeness'],
                        'source_accuracy': scores['source_accuracy'],
                        'overall': scores['overall']
                    }
                })
            else:
                results.append({
                    'pdf': pdf.split('/')[-1], 'success': True, 'items': items,
                    'extract_time': dur, 'scores': None
                })
    
    print(f"  评分完成: {time.time()-t_eval:.1f}s")
    
    # 汇总
    total_time = time.time() - t_extract
    successful = [r for r in results if r.get('success')]
    scored = [r for r in successful if r.get('scores')]
    
    print(f"\n{'='*60}")
    print(f"v7.9 汇总:")
    print(f"  成功率: {len(successful)}/20")
    print(f"  有效评分: {len(scored)}/{len(successful)}")
    
    if scored:
        avg_overall = sum(r['scores']['overall'] for r in scored) / len(scored)
        avg_acc = sum(r['scores']['accuracy'] for r in scored) / len(scored)
        avg_comp = sum(r['scores']['completeness'] for r in scored) / len(scored)
        avg_src = sum(r['scores']['source_accuracy'] for r in scored) / len(scored)
        avg_items = sum(r.get('items', 0) for r in successful) / len(successful)
        avg_time = sum(r.get('extract_time', 0) for r in successful) / len(successful)
        print(f"  平均overall: {avg_overall:.2f}")
        print(f"  平均accuracy: {avg_acc:.2f}")
        print(f"  平均completeness: {avg_comp:.2f}")  
        print(f"  平均source_accuracy: {avg_src:.2f}")
        print(f"  平均条目数: {avg_items:.1f}")
        print(f"  平均提取时间: {avg_time:.1f}s")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    
    with open('v79_20pdf_results.json', 'w') as f:
        json.dump({
            'version': 'v7.9', 'successful': len(successful), 'scored': len(scored),
            'avg_overall': avg_overall if scored else 0,
            'avg_accuracy': avg_acc if scored else 0,
            'avg_completeness': avg_comp if scored else 0,
            'avg_source_accuracy': avg_src if scored else 0,
            'avg_items': avg_items if successful else 0,
            'total_time': total_time, 'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存至: v79_20pdf_results.json")

if __name__ == '__main__':
    main()
