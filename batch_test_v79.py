#!/usr/bin/env python3
"""用同样的20个PDF测试v7.9"""

import sys, json, time
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

def main():
    print(f"\n{'='*60}")
    print(f"测试 v7.9 - 共{len(TEST_PDFS)}个PDF")
    print(f"{'='*60}")
    
    extractor = MedicalPDFExtractor()
    results = []
    total_start = time.time()
    
    for i, pdf_path in enumerate(TEST_PDFS):
        fname = pdf_path.split('/')[-1][:20]
        print(f"[{i+1:2d}/20] {fname}...", end=" ", flush=True)
        
        try:
            t0 = time.time()
            extraction = extractor.extract(pdf_path)
            extract_time = time.time() - t0
        except Exception as e:
            print(f"✗ 异常: {str(e)[:25]}")
            results.append({'pdf': pdf_path.split('/')[-1], 'success': False, 'error': str(e)})
            continue
        
        if not extraction.get('success'):
            print(f"✗ 失败")
            results.append({'pdf': pdf_path.split('/')[-1], 'success': False})
            continue
        
        # v7.9的数据结构可能不同
        data = extraction.get('data', extraction.get('result', {}))
        items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
        
        # GPT评分
        scores = None
        for retry in range(2):
            try:
                scores = evaluate_stable(pdf_path, data, n_runs=1)
                if scores.get('overall', 0) > 0:
                    break
            except:
                time.sleep(2)
        
        if scores and scores.get('overall', 0) > 0:
            overall = scores['overall']
            acc = scores['accuracy']
            comp = scores['completeness']
            src = scores['source_accuracy']
            print(f"✓ {items}条 | acc={acc} comp={comp} src={src} overall={overall}")
            results.append({
                'pdf': pdf_path.split('/')[-1], 'success': True, 'items': items,
                'extract_time': extract_time,
                'scores': {'accuracy': acc, 'completeness': comp, 'source_accuracy': src, 'overall': overall}
            })
        else:
            print(f"✓ {items}条 | 评分失败")
            results.append({'pdf': pdf_path.split('/')[-1], 'success': True, 'items': items, 'extract_time': extract_time, 'scores': None})
    
    total_time = time.time() - total_start
    
    successful = [r for r in results if r.get('success')]
    scored = [r for r in successful if r.get('scores') and r['scores'].get('overall', 0) > 0]
    
    print(f"\n{'-'*60}")
    print(f"v7.9 汇总:")
    print(f"  成功率: {len(successful)}/{len(results)}")
    print(f"  有效评分: {len(scored)}/{len(successful) if successful else 0}")
    
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
            'version': 'v7.9', 'total_pdfs': len(results), 'successful': len(successful),
            'scored': len(scored),
            'avg_overall': avg_overall if scored else 0,
            'avg_accuracy': avg_acc if scored else 0,
            'avg_completeness': avg_comp if scored else 0,
            'avg_source_accuracy': avg_src if scored else 0,
            'avg_items': avg_items if successful else 0,
            'total_time': total_time, 'results': results
        }, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
