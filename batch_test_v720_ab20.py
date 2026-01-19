#!/usr/bin/env python3
"""A/B测试：v7.16 vs v7.20（20个PDF）"""
import sys
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

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
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756709583588_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756710727953_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711072746_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711126015_5579292.pdf",
    "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756711943364_1608702.pdf",
]


def run_version(version_name: str, extractor):
    results = []
    start = time.time()
    scored = 0
    successful = 0

    print(f"\n=== 测试 {version_name} - {len(TEST_PDFS)}个PDF ===")
    for i, pdf_path in enumerate(TEST_PDFS, 1):
        pdf_name = Path(pdf_path).name
        print(f"[{i}/{len(TEST_PDFS)}] {pdf_name}", end=" ", flush=True)
        t0 = time.time()
        try:
            extraction = extractor.extract(pdf_path)
        except Exception as e:
            results.append({"pdf": pdf_name, "success": False, "error": str(e)})
            print(f"✗ 异常: {str(e)[:60]}")
            continue

        if not extraction.get("success"):
            results.append({"pdf": pdf_name, "success": False, "error": extraction.get("error")})
            print(f"✗ 失败: {str(extraction.get('error'))[:60]}")
            continue

        successful += 1
        extract_time = time.time() - t0
        data = extraction.get("result", {})
        items = len(data.get("recommendations", [])) + len(data.get("key_findings", []))

        score = None
        for _ in range(2):
            try:
                score = evaluate_stable(pdf_path, data, n_runs=1)
                break
            except Exception:
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
            print(f"✓ {items}条 | o={score.get('overall'):.1f} a={score.get('accuracy')} c={score.get('completeness')} s={score.get('source_accuracy')}")
        else:
            results.append({
                "pdf": pdf_name,
                "success": True,
                "items": items,
                "extract_time": extract_time,
                "scores": None
            })
            print(f"✓ {items}条 | 评分失败")

    total_time = time.time() - start

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
    print(f"items={summary['avg_items']:.1f} | avg_time={summary['avg_extract_time']:.1f}s | total={summary['total_time']/60:.1f}min")

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

    out_path = Path("/root/pdf_summarization_deploy_20251225_093847/v720_ab20_results.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\n结果已保存: {out_path}")

    # 打印对比
    print("\n========== 20 PDF 对比汇总 ==========")
    print(f"{'指标':<20} {'v7.16':<12} {'v7.20':<12} {'Δ':<10}")
    print("-" * 54)
    for key in ['successful', 'scored', 'avg_overall', 'avg_accuracy', 'avg_completeness', 'avg_source_accuracy', 'avg_items', 'avg_extract_time']:
        v1 = res_716.get(key)
        v2 = res_720.get(key)
        if v1 is None or v2 is None:
            continue
        if isinstance(v1, float):
            delta = v2 - v1
            print(f"{key:<20} {v1:<12.2f} {v2:<12.2f} {delta:+.2f}")
        else:
            delta = v2 - v1
            print(f"{key:<20} {v1:<12} {v2:<12} {delta:+d}")


if __name__ == "__main__":
    main()
