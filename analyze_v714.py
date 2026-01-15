#!/usr/bin/env python3
"""分析v7.14提取结果"""

import sys, json
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v714 import MedicalPDFExtractorV714

pdf_path = "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf"

extractor = MedicalPDFExtractorV714(use_cache=False)
result = extractor.extract(pdf_path)

if result.get('success'):
    data = result.get('result', {})
    print("提取的recommendations:")
    for i, rec in enumerate(data.get('recommendations', [])[:5]):
        print(f"\n{i+1}. {rec.get('text', '')[:100]}...")
        print(f"   sources: {rec.get('sources')}")
        print(f"   quote: {rec.get('original_quote', '')[:80]}...")
else:
    print(f"提取失败: {result.get('error')}")
