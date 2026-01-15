#!/usr/bin/env python3
"""测试v7.13并使用GPT评分"""

import sys, json, time
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v713 import MedicalPDFExtractorV713
from stable_evaluator import evaluate_stable

# 测试PDF列表 (选3个不同的)
TEST_PDFS = [
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756460577061_8106296.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694891048_5579292.pdf",
]

def main():
    extractor = MedicalPDFExtractorV713(use_cache=False)
    
    results = []
    for i, pdf_path in enumerate(TEST_PDFS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_PDFS)}] 测试: {pdf_path.split('/')[-1]}")
        print(f"{'='*60}")
        
        # 提取
        t0 = time.time()
        extraction = extractor.extract(pdf_path)
        extract_time = time.time() - t0
        
        if extraction.get('success'):
            data = extraction.get('result', extraction.get('data', {}))
            items = len(data.get('recommendations', [])) + len(data.get('key_findings', []))
            print(f"  提取成功: {items} 条目, 耗时 {extract_time:.1f}s")
            
            # GPT评分 (n_runs=1 单次评分)
            print(f"  GPT评分中...")
            t1 = time.time()
            scores = evaluate_stable(pdf_path, data, n_runs=1)  # 直接返回分数字典
            eval_time = time.time() - t1
            
            # evaluate_stable 直接返回分数字典
            if scores.get('overall', 0) > 0:
                print(f"  GPT评分结果:")
                print(f"    - accuracy: {scores.get('accuracy', 'N/A')}")
                print(f"    - completeness: {scores.get('completeness', 'N/A')}")
                print(f"    - source_accuracy: {scores.get('source_accuracy', 'N/A')}")
                print(f"    - overall: {scores.get('overall', 'N/A')}")
                if scores.get('details'):
                    reason = scores['details'][0].get('reason', 'N/A') if scores['details'] else 'N/A'
                    print(f"    - reason: {reason}")
                print(f"  评分耗时: {eval_time:.1f}s")
                
                results.append({
                    'pdf': pdf_path.split('/')[-1],
                    'items': items,
                    'extract_time': extract_time,
                    'scores': scores,
                    'eval_time': eval_time
                })
            else:
                print(f"  GPT评分失败: {scores.get('error', 'unknown')}")
                results.append({
                    'pdf': pdf_path.split('/')[-1],
                    'items': items,
                    'extract_time': extract_time,
                    'scores': None,
                    'eval_error': scores.get('error')
                })
        else:
            print(f"  提取失败: {extraction.get('error')}")
    
    # 汇总
    if results:
        print(f"\n{'='*60}")
        print("汇总结果")
        print(f"{'='*60}")
        valid_results = [r for r in results if r.get('scores') and r['scores'].get('overall', 0) > 0]
        if valid_results:
            avg_overall = sum(r['scores']['overall'] for r in valid_results) / len(valid_results)
            avg_acc = sum(r['scores']['accuracy'] for r in valid_results) / len(valid_results)
            avg_comp = sum(r['scores']['completeness'] for r in valid_results) / len(valid_results)
            avg_src = sum(r['scores']['source_accuracy'] for r in valid_results) / len(valid_results)
            print(f"平均overall: {avg_overall:.2f}")
            print(f"平均accuracy: {avg_acc:.2f}")
            print(f"平均completeness: {avg_comp:.2f}")
            print(f"平均source_accuracy: {avg_src:.2f}")
        print(f"平均条目数: {sum(r['items'] for r in results)/len(results):.1f}")
        
        # 保存结果
        with open('/root/pdf_summarization_deploy_20251225_093847/v713_gpt_results.json', 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 v713_gpt_results.json")

if __name__ == '__main__':
    main()
