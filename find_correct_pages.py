#!/usr/bin/env python3
"""找出quote的正确页码"""

import sys, re
import fitz
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v714 import MedicalPDFExtractorV714

pdf_path = "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf"

extractor = MedicalPDFExtractorV714(use_cache=False)
result = extractor.extract(pdf_path)

doc = fitz.open(pdf_path)

def find_page_for_quote(quote: str) -> int:
    """找到引用所在的实际页码"""
    quote_lower = quote[:40].lower().strip()
    for i in range(len(doc)):
        page_text = doc[i].get_text().lower()
        if quote_lower in page_text:
            return i + 1
    return -1

if result.get('success'):
    data = result.get('result', {})
    print("Quote页码分析:\n")
    
    for rec in data.get('recommendations', []):
        quote = rec.get('original_quote', '')
        sources = rec.get('sources', [])
        
        if quote:
            page_match = re.search(r'p(\d+)', str(sources[0]).lower()) if sources else None
            marked_page = int(page_match.group(1)) if page_match else 0
            actual_page = find_page_for_quote(quote)
            
            status = "✓" if marked_page == actual_page else "✗"
            print(f"{status} 标注p{marked_page} 实际p{actual_page}: '{quote[:50]}...'")

doc.close()
