"""v7完整测试脚本"""
import json
import time
import re
import requests
import fitz
from pathlib import Path
from production_extractor_v7 import MedicalPDFExtractor

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
PDF_DIR = Path("/root/autodl-tmp/pdf_summarization_data/pdf_samples")

def is_valid_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    if len(doc) < 2:
        text = doc[0].get_text() if len(doc) > 0 else ""
        doc.close()
        return len(text) > 100 and "下载链接已失效" not in text
    doc.close()
    return True

def evaluate(result, doc_type):
    prompt = f"""评估{doc_type}医学文档提取质量，5个维度(1-10分):
1. accuracy: 信息准确性
2. completeness: 信息完整性  
3. structure: 结构合理性
4. page_accuracy: 页码标注准确性
5. no_hallucination: 无编造(是否只含原文信息)

提取结果：
```json
{json.dumps(result, ensure_ascii=False, indent=2)[:4000]}
```

返回JSON: {{"accuracy":X,"completeness":X,"structure":X,"page_accuracy":X,"no_hallucination":X,"average":X,"issues":"存在的问题(简述)"}}"""
    
    resp = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"},
        json={"model": "gpt-4.1", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500}, timeout=90)
    match = re.search(r'\{.*\}', resp.json()["choices"][0]["message"]["content"], re.DOTALL)
    return json.loads(match.group()) if match else {"average": 0, "issues": "评估失败"}

def main():
    extractor = MedicalPDFExtractor()
    pdfs = [p for p in sorted(PDF_DIR.glob("*.pdf")) if is_valid_pdf(str(p))]
    
    print("=" * 70)
    print("v7 动态Few-shot 完整测试")
    print("=" * 70)
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        
        ext = extractor.extract(str(pdf))
        if ext["success"]:
            ev = evaluate(ext["result"], ext["doc_type"])
            score = ev.get("average", 0)
            issues = ev.get("issues", "")[:60]
            print(f"  类型: {ext['doc_type']}, 耗时: {ext['time']:.0f}s, 评分: {score}/10")
            if score < 8:
                print(f"  问题: {issues}")
            results.append({
                "pdf": pdf.name, 
                "doc_type": ext["doc_type"], 
                "score": score, 
                "time": ext["time"],
                "eval": ev,
                "extraction": ext["result"]
            })
        else:
            print(f"  失败: {ext.get('error', 'Unknown')[:50]}")
            results.append({"pdf": pdf.name, "success": False, "error": ext.get("error")})
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    successful = [r for r in results if "score" in r]
    scores = [r["score"] for r in successful]
    times = [r["time"] for r in successful]
    
    print(f"成功率: {len(successful)}/{len(results)}")
    print(f"平均分: {sum(scores)/len(scores):.2f}/10")
    print(f"平均耗时: {sum(times)/len(times):.0f}s")
    print(f"≥8分占比: {len([s for s in scores if s >= 8])/len(scores)*100:.0f}%")
    print(f"≥9分占比: {len([s for s in scores if s >= 9])/len(scores)*100:.0f}%")
    
    print(f"\n各维度平均:")
    dims = ["accuracy", "completeness", "structure", "page_accuracy", "no_hallucination"]
    for d in dims:
        avg = sum(r["eval"].get(d, 0) for r in successful) / len(successful)
        print(f"  {d}: {avg:.1f}")
    
    print(f"\n详细分数:")
    for r in successful:
        status = "✓" if r["score"] >= 8 else "✗"
        print(f"  {status} {r['pdf'][:25]}... {r['doc_type']:10} {r['score']}/10")
    
    # 保存
    with open("v7_test_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 v7_test_results.json")

if __name__ == "__main__":
    main()
