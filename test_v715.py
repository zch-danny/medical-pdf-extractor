#!/usr/bin/env python3
"""测试v7.15"""

import sys, json, time
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v715 import MedicalPDFExtractorV715
from stable_evaluator import evaluate_stable

TEST_PDFS = [
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460577061_8106296.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694891048_5579292.pdf",
]

def main():
    extractor = MedicalPDFExtractorV715(use_cache=False)
    
    results = []
    for i, pdf_path in enumerate(TEST_PDFS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_PDFS)}] {pdf_path.split('/')[-1]}")
        print(f"{'='*60}")
        
        t0 = time.time()
        extraction = extractor.extract(pdf_path)
        extract_time = time.time() - t0
        
        if extraction.get('success'):
            data = extraction.get('result', {})
            items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
            verify_stats = extraction.get('stats', {}).get('source_verification', {})
            print(f"  提取: {items} 条目, {extract_time:.1f}s")
            print(f"  验证: verified={verify_stats.get('verified',0)}, fixed={verify_stats.get('fixed',0)}, removed={verify_stats.get('removed',0)}")
            
            # GPT评分
            print(f"  评分中...")
            try:
                scores = evaluate_stable(pdf_path, data, n_runs=1)
                if scores.get('overall', 0) > 0:
                    print(f"  GPT: acc={scores['accuracy']}, comp={scores['completeness']}, src={scores['source_accuracy']}, overall={scores['overall']}")
                    if scores.get('details'):
                        print(f"  原因: {scores['details'][0].get('reason', '')[:80]}...")
                    results.append({'pdf': pdf_path.split('/')[-1], 'items': items, 'verify': verify_stats, 'scores': scores})
                else:
                    print(f"  评分失败")
            except Exception as e:
                print(f"  评分异常: {e}")
        else:
            print(f"  提取失败: {extraction.get('error')}")
    
    if results:
        print(f"\n{'='*60}")
        print("v7.15 汇总")
        print(f"{'='*60}")
        valid = [r for r in results if r.get('scores', {}).get('overall', 0) > 0]
        if valid:
            print(f"平均overall: {sum(r['scores']['overall'] for r in valid)/len(valid):.2f}")
            print(f"平均source_accuracy: {sum(r['scores']['source_accuracy'] for r in valid)/len(valid):.2f}")
        print(f"平均条目数: {sum(r['items'] for r in results)/len(results):.1f}")

if __name__ == '__main__':
    main()
