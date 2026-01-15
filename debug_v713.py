#!/usr/bin/env python3
"""Debug v7.13 extraction"""

import sys, json
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from production_extractor_v713 import MedicalPDFExtractorV713

pdf_path = "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453705743_8248372.pdf"

extractor = MedicalPDFExtractorV713(use_cache=False)
result = extractor.extract(pdf_path)

print("Full result:")
print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
