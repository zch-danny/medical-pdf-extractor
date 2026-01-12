"""批量测试方案5 v4"""
import os
import json
import time
from pathlib import Path
from field_retrieval_extractor import extract_pdf
import fitz

PDF_DIR = "/root/autodl-tmp/pdf_input/pdf_input_09-1125"

def get_page_count(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except:
        return 0

def run_batch_test(max_files=15):
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))[:max_files]
    
    results = []
    total_start = time.time()
    
    print(f"开始测试 {len(pdf_files)} 个PDF...")
    print("="*70)
    
    for i, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name[:20]
        page_count = get_page_count(str(pdf_path))
        
        print(f"\n[{i+1}/{len(pdf_files)}] {pdf_name}... ({page_count}页)")
        
        try:
            result = extract_pdf(str(pdf_path))
            
            stats = result.get("stats", {})
            results.append({
                "file": pdf_path.name,
                "pages": page_count,
                "success": result.get("success"),
                "doc_type": result.get("doc_type"),
                "time": result.get("time", 0),
                "coverage": stats.get("coverage_ratio", 0),
                "covered_pages": len(stats.get("covered_pages", [])),
                "findings": stats.get("findings_count", 0),
                "conclusions": stats.get("conclusions_count", 0),
                "recommendations": stats.get("recommendations_count", 0),
                "toc_detected": len(stats.get("toc_pages_detected", [])),
            })
            
            status = "✓" if result.get("success") else "✗"
            print(f"  {status} 类型:{result.get('doc_type')} 覆盖:{stats.get('coverage_ratio',0)*100:.1f}% "
                  f"发现:{stats.get('findings_count',0)} 耗时:{result.get('time',0):.1f}s")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            results.append({
                "file": pdf_path.name,
                "pages": page_count,
                "success": False,
                "error": str(e)
            })
    
    total_time = time.time() - total_start
    
    # 统计
    print("\n" + "="*70)
    print("汇总统计")
    print("="*70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"成功率: {len(successful)}/{len(results)} ({len(successful)*100//len(results)}%)")
    print(f"总耗时: {total_time:.1f}s")
    print(f"平均耗时: {total_time/len(results):.1f}s/文件")
    
    if successful:
        avg_coverage = sum(r["coverage"] for r in successful) / len(successful)
        avg_findings = sum(r["findings"] for r in successful) / len(successful)
        avg_conclusions = sum(r["conclusions"] for r in successful) / len(successful)
        toc_detected = sum(1 for r in successful if r.get("toc_detected", 0) > 0)
        
        print(f"\n平均覆盖率: {avg_coverage*100:.1f}%")
        print(f"平均发现数: {avg_findings:.1f}")
        print(f"平均结论数: {avg_conclusions:.1f}")
        print(f"检测到目录的文档: {toc_detected}/{len(successful)}")
        
        # 按类型统计
        types = {}
        for r in successful:
            t = r.get("doc_type", "UNKNOWN")
            if t not in types:
                types[t] = []
            types[t].append(r)
        
        print(f"\n按类型统计:")
        for t, docs in types.items():
            avg_cov = sum(d["coverage"] for d in docs) / len(docs)
            print(f"  {t}: {len(docs)}个, 平均覆盖率{avg_cov*100:.1f}%")
    
    # 保存结果
    output_file = "/root/pdf_summarization_deploy_20251225_093847/v4_batch_test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": {
            "total": len(results),
            "success": len(successful),
            "failed": len(failed),
            "total_time": total_time,
            "avg_coverage": avg_coverage if successful else 0
        }}, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    return results

if __name__ == "__main__":
    run_batch_test(15)
