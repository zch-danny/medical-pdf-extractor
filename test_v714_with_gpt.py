#!/usr/bin/env python3
"""测试v7.14并使用GPT评分"""

import sys, json, time
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v714 import MedicalPDFExtractorV714
from stable_evaluator import evaluate_stable

TEST_PDFS = [
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460577061_8106296.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694891048_5579292.pdf",
]

def main():
    extractor = MedicalPDFExtractorV714(use_cache=False)
    
    results = []
    for i, pdf_path in enumerate(TEST_PDFS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_PDFS)}] 测试: {pdf_path.split('/')[-1]}")
        print(f"{'='*60}")
        
        t0 = time.time()
        extraction = extractor.extract(pdf_path)
        extract_time = time.time() - t0
        
        if extraction.get('success'):
            data = extraction.get('result', {})
            items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
            verify_stats = extraction.get('stats', {}).get('source_verification', {})
            print(f"  提取成功: {items} 条目, 耗时 {extract_time:.1f}s")
            print(f"  Source验证: {verify_stats}")
            
            print(f"  GPT评分中...")
            t1 = time.time()
            scores = evaluate_stable(pdf_path, data, n_runs=1)
            eval_time = time.time() - t1
            
            if scores.get('overall', 0) > 0:
                print(f"  GPT评分结果:")
                print(f"    - accuracy: {scores.get('accuracy')}")
                print(f"    - completeness: {scores.get('completeness')}")
                print(f"    - source_accuracy: {scores.get('source_accuracy')}")
                print(f"    - overall: {scores.get('overall')}")
                if scores.get('details'):
                    print(f"    - reason: {scores['details'][0].get('reason', 'N/A')}")
                print(f"  评分耗时: {eval_time:.1f}s")
                
                results.append({
                    'pdf': pdf_path.split('/')[-1],
                    'items': items,
                    'extract_time': extract_time,
                    'verify_stats': verify_stats,
                    'scores': scores,
                })
            else:
                print(f"  GPT评分失败: {scores.get('error')}")
        else:
            print(f"  提取失败: {extraction.get('error')}")
    
    if results:
        print(f"\n{'='*60}")
        print("v7.14 汇总结果")
        print(f"{'='*60}")
        valid = [r for r in results if r.get('scores', {}).get('overall', 0) > 0]
        if valid:
            print(f"平均overall: {sum(r['scores']['overall'] for r in valid)/len(valid):.2f}")
            print(f"平均accuracy: {sum(r['scores']['accuracy'] for r in valid)/len(valid):.2f}")
            print(f"平均completeness: {sum(r['scores']['completeness'] for r in valid)/len(valid):.2f}")
            print(f"平均source_accuracy: {sum(r['scores']['source_accuracy'] for r in valid)/len(valid):.2f}")
        print(f"平均条目数: {sum(r['items'] for r in results)/len(results):.1f}")
        
        with open('/root/pdf_summarization_deploy_20251225_093847/v714_gpt_results.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 v714_gpt_results.json")

if __name__ == '__main__':
    main()
