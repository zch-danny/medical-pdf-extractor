#!/usr/bin/env python3
"""验证v7.14页码准确性"""

import sys, json, re
import fitz
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v714 import MedicalPDFExtractorV714

pdf_path = "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf"

# 提取结果
extractor = MedicalPDFExtractorV714(use_cache=False)
result = extractor.extract(pdf_path)

# 打开PDF
doc = fitz.open(pdf_path)

def check_quote_in_page(quote: str, page_num: int) -> bool:
    """检查引用是否在指定页面"""
    if page_num < 1 or page_num > len(doc):
        return False
    page_text = doc[page_num - 1].get_text().lower()
    # 取引用前20字检查
    quote_start = quote[:50].lower().strip()
    return quote_start[:30] in page_text

if result.get('success'):
    data = result.get('result', {})
    print("验证页码准确性:")
    correct = 0
    total = 0
    
    for rec in data.get('recommendations', []):
        quote = rec.get('original_quote', '')
        sources = rec.get('sources', [])
        
        if quote and sources:
            total += 1
            page_match = re.search(r'p(\d+)', str(sources[0]).lower())
            if page_match:
                page_num = int(page_match.group(1))
                is_correct = check_quote_in_page(quote, page_num)
                status = "✓" if is_correct else "✗"
                if is_correct:
                    correct += 1
                print(f"{status} p{page_num}: '{quote[:60]}...'")
    
    print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")

doc.close()
