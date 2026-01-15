#!/usr/bin/env python3
"""批量测试v7.16 - 增强错误处理"""

import sys, json, time
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v716 import MedicalPDFExtractorV716
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
    print(f"测试 v7.16 - 共{len(TEST_PDFS)}个PDF (增强版)")
    print(f"{'='*60}")
    
    extractor = MedicalPDFExtractorV716(use_cache=False)
    results = []
    total_start = time.time()
    
    for i, pdf_path in enumerate(TEST_PDFS):
        fname = pdf_path.split('/')[-1][:20]
        print(f"[{i+1:2d}/20] {fname}...", end=" ", flush=True)
        
        # 提取
        try:
            t0 = time.time()
            extraction = extractor.extract(pdf_path)
            extract_time = time.time() - t0
        except Exception as e:
            print(f"✗ 提取异常: {str(e)[:25]}")
            results.append({'pdf': pdf_path.split('/')[-1], 'success': False, 'error': str(e)})
            continue
        
        if not extraction.get('success'):
            print(f"✗ 提取失败")
            results.append({'pdf': pdf_path.split('/')[-1], 'success': False, 'error': extraction.get('error')})
            continue
        
        data = extraction.get('result', {})
        items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
        
        # GPT评分 (带重试)
        scores = None
        for retry in range(2):
            try:
                scores = evaluate_stable(pdf_path, data, n_runs=1)
                if scores.get('overall', 0) > 0:
                    break
            except Exception as e:
                if retry == 0:
                    time.sleep(2)  # 重试前等待
                continue
        
        if scores and scores.get('overall', 0) > 0:
            overall = scores.get('overall', 0)
            acc = scores.get('accuracy', 0)
            comp = scores.get('completeness', 0)
            src = scores.get('source_accuracy', 0)
            print(f"✓ {items}条 | acc={acc} comp={comp} src={src} overall={overall}")
            results.append({
                'pdf': pdf_path.split('/')[-1], 'success': True, 'items': items,
                'extract_time': extract_time,
                'scores': {'accuracy': acc, 'completeness': comp, 'source_accuracy': src, 'overall': overall}
            })
        else:
            print(f"✓ {items}条 | 评分失败")
            results.append({
                'pdf': pdf_path.split('/')[-1], 'success': True, 'items': items,
                'extract_time': extract_time, 'scores': None
            })
    
    total_time = time.time() - total_start
    
    # 统计
    successful = [r for r in results if r.get('success')]
    scored = [r for r in successful if r.get('scores') and r['scores'].get('overall', 0) > 0]
    
    print(f"\n{'-'*60}")
    print(f"v7.16 汇总:")
    print(f"  成功率: {len(successful)}/{len(results)}")
    print(f"  有效评分: {len(scored)}/{len(successful) if successful else 0}")
    
    avg_overall = avg_acc = avg_comp = avg_src = avg_items = avg_time = 0
    if scored:
        avg_overall = sum(r['scores']['overall'] for r in scored) / len(scored)
        avg_acc = sum(r['scores']['accuracy'] for r in scored) / len(scored)
        avg_comp = sum(r['scores']['completeness'] for r in scored) / len(scored)
        avg_src = sum(r['scores']['source_accuracy'] for r in scored) / len(scored)
        print(f"  平均overall: {avg_overall:.2f}")
        print(f"  平均accuracy: {avg_acc:.2f}")
        print(f"  平均completeness: {avg_comp:.2f}")
        print(f"  平均source_accuracy: {avg_src:.2f}")
    if successful:
        avg_items = sum(r.get('items', 0) for r in successful) / len(successful)
        avg_time = sum(r.get('extract_time', 0) for r in successful) / len(successful)
        print(f"  平均条目数: {avg_items:.1f}")
        print(f"  平均提取时间: {avg_time:.1f}s")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    
    # 保存
    with open('/root/pdf_summarization_deploy_20251225_093847/v716_20pdf_results.json', 'w') as f:
        json.dump({
            'version': 'v7.16', 'total_pdfs': len(results), 'successful': len(successful),
            'scored': len(scored), 'avg_overall': avg_overall, 'avg_accuracy': avg_acc,
            'avg_completeness': avg_comp, 'avg_source_accuracy': avg_src,
            'avg_items': avg_items, 'total_time': total_time, 'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"  结果保存至: v716_20pdf_results.json")

if __name__ == '__main__':
    main()
