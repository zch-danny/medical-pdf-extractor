#!/usr/bin/env python3
"""
PDF Sample Batch Test Script
Test all 10 sample PDFs against the summarization API

Usage:
    python test_samples.py [--api-url http://localhost:8080]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Please install httpx: pip install httpx")
    sys.exit(1)


PDF_SAMPLES = [
    ("1756965249653_1608702.pdf", "2KB", "tiny"),
    ("1760653051590_4754896.pdf", "63KB", "small"),
    ("1756864558155_1608702.pdf", "140KB", "small"),
    ("1760596671555_5579292.pdf", "335KB", "medium-small"),
    ("1758019170505_5579292.pdf", "338KB", "medium-small"),
    ("1760091187079_1608702.pdf", "927KB", "medium"),
    ("1759146059792_8106296.pdf", "1047KB", "~1MB"),
    ("1757066770930_1608702.pdf", "2356KB", "large"),
    ("1761058889527_1608702.pdf", "3308KB", "large"),
    ("1761185421862_1608702.pdf", "13890KB", "very-large"),
]


def test_pdf(client: httpx.Client, api_url: str, pdf_path: Path, timeout: float = 300.0):
    """Test single PDF file"""
    url = f"{api_url}/api/v1/summarize"
    
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {"language": "zh", "max_length": 500}
        
        start_time = time.time()
        try:
            response = client.post(url, files=files, data=data, timeout=timeout)
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "status_code": 200,
                    "elapsed": round(elapsed, 2),
                    "summary_length": len(result.get("data", {}).get("summary", "")),
                    "word_count": result.get("data", {}).get("word_count", 0),
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "elapsed": round(elapsed, 2),
                    "error": response.text[:200],
                }
        except httpx.TimeoutException:
            return {
                "success": False,
                "status_code": 0,
                "elapsed": timeout,
                "error": "Request timeout",
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "elapsed": time.time() - start_time,
                "error": str(e)[:200],
            }


def main():
    parser = argparse.ArgumentParser(description="Test PDF samples")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds")
    parser.add_argument("--samples-dir", default="pdf_samples", help="PDF samples directory")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)

    print("=" * 70)
    print("PDF Summarization API - Sample Test")
    print("=" * 70)
    print(f"API URL: {args.api_url}")
    print(f"Timeout: {args.timeout}s")
    print(f"Samples: {samples_dir.absolute()}")
    print("=" * 70)

    # Check API health
    print("\n[0] Checking API health...")
    try:
        with httpx.Client() as client:
            resp = client.get(f"{args.api_url}/health", timeout=10)
            if resp.status_code == 200:
                print("    API is healthy!")
            else:
                print(f"    Warning: Health check returned {resp.status_code}")
    except Exception as e:
        print(f"    Error: Cannot connect to API - {e}")
        sys.exit(1)

    # Test each PDF
    results = []
    success_count = 0
    total_time = 0

    with httpx.Client() as client:
        for i, (filename, size, desc) in enumerate(PDF_SAMPLES, 1):
            pdf_path = samples_dir / filename
            
            print(f"\n[{i}/10] Testing: {filename}")
            print(f"       Size: {size} ({desc})")
            
            if not pdf_path.exists():
                print(f"       SKIP: File not found")
                results.append({"file": filename, "success": False, "error": "File not found"})
                continue

            result = test_pdf(client, args.api_url, pdf_path, args.timeout)
            result["file"] = filename
            result["size"] = size
            result["desc"] = desc
            results.append(result)

            if result["success"]:
                success_count += 1
                total_time += result["elapsed"]
                print(f"       OK: {result['elapsed']}s, summary={result['summary_length']} chars")
            else:
                print(f"       FAIL: {result.get('error', 'Unknown error')[:50]}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total:   {len(PDF_SAMPLES)} PDFs")
    print(f"Success: {success_count}/{len(PDF_SAMPLES)}")
    print(f"Failed:  {len(PDF_SAMPLES) - success_count}/{len(PDF_SAMPLES)}")
    if success_count > 0:
        print(f"Avg Time: {round(total_time / success_count, 2)}s per PDF")
        print(f"Total Time: {round(total_time, 2)}s")

    # Save results
    results_file = "test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "api_url": args.api_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_count": success_count,
            "total_count": len(PDF_SAMPLES),
            "total_time": round(total_time, 2),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")

    # Exit code
    sys.exit(0 if success_count == len(PDF_SAMPLES) else 1)


if __name__ == "__main__":
    main()
