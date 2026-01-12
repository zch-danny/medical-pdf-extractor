"""
对比测试：动态Few-shot vs v6 baseline
"""
import json
import os
import time
import requests
from pathlib import Path
from dynamic_fewshot_extractor import extract_with_fewshot

# GPT评估API
GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

PDF_DIR = Path("/root/autodl-tmp/pdf_summarization_data/pdf_samples")

def evaluate_extraction(extraction_result: dict, doc_type: str) -> dict:
    """使用GPT-5.2评估提取质量"""
    prompt = f"""你是医学文献信息提取质量评估专家。请评估以下{doc_type}类型文档的提取结果。

提取结果：
```json
{json.dumps(extraction_result, ensure_ascii=False, indent=2)[:4000]}
```

请从以下5个维度评分（1-10分）：
1. accuracy: 信息准确性
2. completeness: 信息完整性
3. structure: 结构合理性
4. page_accuracy: 页码标注准确性
5. no_hallucination: 无编造（是否只包含原文信息）

返回JSON格式：
{{"accuracy": X, "completeness": X, "structure": X, "page_accuracy": X, "no_hallucination": X, "average": X, "brief_comment": "简短评价"}}

只返回JSON。"""

    resp = requests.post(
        GPT_API,
        headers={"Authorization": f"Bearer {GPT_KEY}"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.2},
        timeout=60
    )
    content = resp.json()["choices"][0]["message"]["content"]
    
    import re
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"error": "评估解析失败"}

def main():
    # 获取测试PDF列表
    pdfs = sorted(PDF_DIR.glob("*.pdf"))[:5]  # 先测5个
    
    print("=" * 60)
    print("动态Few-shot提取器测试")
    print("=" * 60)
    
    results = []
    total_time = 0
    total_score = 0
    
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        
        start = time.time()
        extraction = extract_with_fewshot(str(pdf))
        elapsed = time.time() - start
        total_time += elapsed
        
        if extraction["success"]:
            print(f"  分类: {extraction['doc_type']}, Few-shot: {extraction['has_fewshot']}, 耗时: {elapsed:.1f}s")
            
            # 评估
            eval_result = evaluate_extraction(extraction["result"], extraction["doc_type"])
            avg_score = eval_result.get("average", 0)
            total_score += avg_score
            
            print(f"  评分: {avg_score}/10 | {eval_result.get('brief_comment', '')[:50]}")
            
            results.append({
                "pdf": pdf.name,
                "doc_type": extraction["doc_type"],
                "has_fewshot": extraction["has_fewshot"],
                "time": elapsed,
                "score": avg_score,
                "eval": eval_result
            })
        else:
            print(f"  失败: {extraction.get('error', 'Unknown')}")
            results.append({
                "pdf": pdf.name,
                "success": False,
                "error": extraction.get("error")
            })
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    successful = [r for r in results if "score" in r]
    if successful:
        avg_score = sum(r["score"] for r in successful) / len(successful)
        avg_time = sum(r["time"] for r in successful) / len(successful)
        print(f"成功率: {len(successful)}/{len(results)}")
        print(f"平均分: {avg_score:.1f}/10")
        print(f"平均耗时: {avg_time:.1f}s")
        print(f"\n各PDF得分:")
        for r in results:
            if "score" in r:
                print(f"  {r['pdf']}: {r['score']}/10 ({r['doc_type']})")
    
    # 保存结果
    with open("fewshot_test_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 fewshot_test_results.json")

if __name__ == "__main__":
    main()
