#!/usr/bin/env python3
"""
生产级医学PDF结构化提取器 v7.12
基于v7.11，添加后处理页码修正(不影响LLM提取)
"""

import os, sys, json, re, time
from pathlib import Path
from typing import Dict, List, Optional
import fitz
import requests

# 直接继承v7.11
from production_extractor_v711_quote import MedicalPDFExtractorV711

class MedicalPDFExtractorV712(MedicalPDFExtractorV711):
    """v7.12 = v7.11 + 后处理页码修正"""
    
    def _fix_sources_postprocess(self, result: Dict, pages_text: Dict[int, str]) -> Dict:
        """后处理: 校验并修正错误的页码标注"""
        def norm(s):
            return re.sub(r'\s+', ' ', (s or '').strip()).lower()
        
        def find_best_page(text: str) -> Optional[int]:
            text_norm = norm(text)
            words = [w for w in text_norm.split() if len(w) > 3][:15]
            if len(words) < 3:
                return None
            
            best_page, best_score = None, 0
            for page_num, page_text in pages_text.items():
                page_norm = norm(page_text)
                match_count = sum(1 for w in words if w in page_norm)
                if match_count > best_score and match_count >= max(3, len(words) * 0.25):
                    best_score = match_count
                    best_page = page_num
            
            return best_page
        
        def parse_source_page(sources: List) -> Optional[int]:
            for s in (sources or []):
                m = re.search(r'p(\d+)', str(s), re.I)
                if m:
                    return int(m.group(1))
            return None
        
        def verify_source(text: str, sources: List) -> bool:
            page_num = parse_source_page(sources)
            if not page_num or page_num not in pages_text:
                return False
            
            text_norm = norm(text)
            page_norm = norm(pages_text[page_num])
            words = [w for w in text_norm.split() if len(w) > 3][:10]
            if len(words) < 2:
                return True
            
            match_count = sum(1 for w in words if w in page_norm)
            return match_count >= max(2, len(words) * 0.25)
        
        # 只修正错误标注，不删除条目
        for rec in result.get('recommendations', []):
            text = rec.get('text', '')
            if text and not verify_source(text, rec.get('sources', [])):
                best_page = find_best_page(text)
                if best_page:
                    rec['sources'] = [f"p{best_page}"]
        
        for finding in result.get('key_findings', []):
            text = finding.get('finding', '')
            if text and not verify_source(text, finding.get('sources', [])):
                best_page = find_best_page(text)
                if best_page:
                    finding['sources'] = [f"p{best_page}"]
        
        for ev in result.get('key_evidence', []):
            text = ev.get('evidence', '')
            if text and not verify_source(text, ev.get('sources', [])):
                best_page = find_best_page(text)
                if best_page:
                    ev['sources'] = [f"p{best_page}"]
        
        return result
    
    def extract(self, pdf_path: str) -> Dict:
        """调用v7.11的extract，然后添加后处理"""
        start_time = time.time()
        
        # 先用v7.11提取
        result = super().extract(pdf_path)
        
        if not result.get('success'):
            result['version'] = 'v7.12'
            return result
        
        # 获取页面文本用于后处理
        try:
            doc = fitz.open(pdf_path)
            pages_text = {i+1: doc[i].get_text() for i in range(len(doc))}
            doc.close()
            
            # 后处理修正页码
            result['result'] = self._fix_sources_postprocess(result['result'], pages_text)
        except:
            pass
        
        result['version'] = 'v7.12'
        result['time'] = time.time() - start_time
        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python production_extractor_v712.py <pdf_path>")
        sys.exit(1)
    
    extractor = MedicalPDFExtractorV712()
    result = extractor.extract(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
