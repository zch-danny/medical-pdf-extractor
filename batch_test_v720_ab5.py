#!/usr/bin/env python3
"""A/B测试：v7.16 vs v7.20（5个PDF）"""
import time
import json
from pathlib import Path

from production_extractor_v716 import MedicalPDFExtractorV716
from production_extractor_v720 import MedicalPDFExtractorV720
from stable_evaluator import evaluate_stable

TEST_PDFS = [
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711187745_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711584225_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453891288_1608702.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694891048_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756694954932_5579292.pdf",
]


def run_version(version_name: str, extractor):
    results = []
    start = time.time()
    scored = 0
    successful = 0

    print(f"\n=== 测试 {version_name} - {len(TEST_PDFS)}个PDF ===")
    for i, pdf_path in enumerate(TEST_PDFS, 1):
        pdf_name = Path(pdf_path).name
        print(f"[{i}/{len(TEST_PDFS)}] {pdf_name}")
        t0 = time.time()
        try:
            extraction = extractor.extract(pdf_path)
        except Exception as e:
            results.append({"pdf": pdf_name, "success": False, "error": str(e)})
            print(f"  ✗ 提取异常: {e}")
            continue

        if not extraction.get("success"):
            results.append({"pdf": pdf_name, "success": False, "error": extraction.get("error")})
            print(f"  ✗ 提取失败: {extraction.get('error')}")
            continue

        successful += 1
        extract_time = time.time() - t0
        data = extraction.get("result", {})
        items = len(data.get("recommendations", [])) + len(data.get("key_findings", []))

        # 评分（1次）
        score = None
        for _ in range(2):
            try:
                score = evaluate_stable(pdf_path, data, n_runs=1)
                break
            except Exception as e:
                last_err = str(e)
                score = None
        if score and score.get("overall", 0) > 0:
            scored += 1
            results.append({
                "pdf": pdf_name,
                "success": True,
                "items": items,
                "extract_time": extract_time,
                "scores": {
                    "overall": score.get("overall"),
                    "accuracy": score.get("accuracy"),
                    "completeness": score.get("completeness"),
                    "source_accuracy": score.get("source_accuracy"),
                }
            })
            print(f"  ✓ {items}条 | acc={score.get('accuracy')} comp={score.get('completeness')} src={score.get('source_accuracy')} overall={score.get('overall')}")
        else:
            results.append({
                "pdf": pdf_name,
                "success": True,
                "items": items,
                "extract_time": extract_time,
                "scores": None
            })
            print(f"  ✓ {items}条 | 评分失败")

    total_time = time.time() - start

    # 汇总
    scored_items = [r for r in results if r.get("scores")]
    avg = {}
    if scored_items:
        avg["overall"] = sum(r["scores"]["overall"] for r in scored_items) / len(scored_items)
        avg["accuracy"] = sum(r["scores"]["accuracy"] for r in scored_items) / len(scored_items)
        avg["completeness"] = sum(r["scores"]["completeness"] for r in scored_items) / len(scored_items)
        avg["source_accuracy"] = sum(r["scores"]["source_accuracy"] for r in scored_items) / len(scored_items)
    avg_items = sum(r.get("items", 0) for r in results if r.get("success")) / max(1, successful)
    avg_time = sum(r.get("extract_time", 0) for r in results if r.get("success")) / max(1, successful)

    summary = {
        "version": version_name,
        "total_pdfs": len(TEST_PDFS),
        "successful": successful,
        "scored": scored,
        "avg_overall": avg.get("overall"),
        "avg_accuracy": avg.get("accuracy"),
        "avg_completeness": avg.get("completeness"),
        "avg_source_accuracy": avg.get("source_accuracy"),
        "avg_items": avg_items,
        "avg_extract_time": avg_time,
        "total_time": total_time,
        "details": results
    }

    print(f"\n--- {version_name} 汇总 ---")
    print(f"成功: {successful}/{len(TEST_PDFS)} | 评分: {scored}/{len(TEST_PDFS)}")
    if scored_items:
        print(f"overall={summary['avg_overall']:.2f} acc={summary['avg_accuracy']:.2f} comp={summary['avg_completeness']:.2f} src={summary['avg_source_accuracy']:.2f}")
    print(f"items={summary['avg_items']:.1f} | avg_time={summary['avg_extract_time']:.1f}s | total={summary['total_time']:.1f}s")

    return summary


def main():
    v716 = MedicalPDFExtractorV716(use_cache=False)
    v720 = MedicalPDFExtractorV720(use_cache=False)

    res_716 = run_version("v7.16", v716)
    res_720 = run_version("v7.20", v720)

    output = {
        "compare": {
            "v7.16": res_716,
            "v7.20": res_720
        }
    }

    out_path = Path("/root/pdf_summarization_deploy_20251225_093847/v720_ab5_results.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
