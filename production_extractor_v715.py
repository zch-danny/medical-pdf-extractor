#!/usr/bin/env python3
"""
医学PDF结构化提取器 v7.15
核心改进: 强制后处理source修正
- 基于quote精确匹配修正页码
- 删除无法验证的条目
"""

import os, sys, json, re, time, hashlib, sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz
import requests
from concurrent.futures import ThreadPoolExecutor

class ExtractionCache:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "extraction_cache.db"
        self.db_path = str(db_path)
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                pdf_hash TEXT PRIMARY KEY,
                result TEXT,
                version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_pdf_hash(self, pdf_path: str) -> str:
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get(self, pdf_hash: str, version: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT result FROM cache WHERE pdf_hash = ? AND version = ?',
            (pdf_hash, version)
        )
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    
    def set(self, pdf_hash: str, result: Dict, version: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            'INSERT OR REPLACE INTO cache (pdf_hash, result, version) VALUES (?, ?, ?)',
            (pdf_hash, json.dumps(result, ensure_ascii=False), version)
        )
        conn.commit()
        conn.close()
    
    def clear(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM cache')
        conn.commit()
        conn.close()


class MedicalPDFExtractorV715:
    VERSION = "v7.15"
    
    SKIP_PATTERNS = [
        r'^(table of contents|contents|目录)',
        r'^(references?|bibliography|参考文献)',
        r'^(appendix|附录)',
    ]
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions", 
                 use_cache: bool = True, cache_path: str = None):
        self.api_url = api_url
        self.model = "qwen3-8b"
        self.use_cache = use_cache
        self.cache = ExtractionCache(cache_path) if use_cache else None
    
    def _call_llm(self, prompt: str, max_tokens: int = 2000, json_mode: bool = False) -> Tuple[str, Dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        resp = requests.post(self.api_url, json=payload, timeout=180)
        result = resp.json()
        return result["choices"][0]["message"]["content"], result.get("usage", {})
    
    def _extract_json(self, text: str) -> Dict:
        """从响应中提取JSON，处理可能的截断"""
        # 直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试修复截断的JSON
        # 找到recommendations或key_findings数组
        for key in ['recommendations', 'key_findings']:
            pattern = rf'"{key}"\s*:\s*\['
            match = re.search(pattern, text)
            if match:
                start = text.find('{')
                # 找最后一个完整的对象 }]
                last_complete = text.rfind('}]')
                if last_complete > 0:
                    truncated = text[start:last_complete+2] + '}'
                    try:
                        return json.loads(truncated)
                    except:
                        pass
                # 尝试找最后一个}并补全
                last_brace = text.rfind('}')
                if last_brace > start:
                    # 计算缺少的括号
                    substr = text[start:last_brace+1]
                    open_brackets = substr.count('[') - substr.count(']')
                    open_braces = substr.count('{') - substr.count('}')
                    suffix = ']' * open_brackets + '}' * open_braces
                    try:
                        return json.loads(substr + suffix)
                    except:
                        pass
        
        # 回退到正则匹配
        for pat in [r'```json\s*([\s\S]*?)\s*```', r'\{[\s\S]*\}']:
            match = re.search(pat, text)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pat else match.group())
                except:
                    continue
        return {}
    
    def _should_skip_page(self, text: str) -> bool:
        first_line = text.strip().split('\n')[0].lower() if text.strip() else ""
        return any(re.match(p, first_line, re.I) for p in self.SKIP_PATTERNS)
    
    def _detect_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['guideline', 'recommendation', '指南', '推荐']):
            return "GUIDELINE"
        if any(kw in text_lower for kw in ['systematic review', 'meta-analysis', '系统综述']):
            return "REVIEW"
        return "OTHER"
    
    def _extract_content(self, pdf_path: str) -> Tuple[Dict, Dict, Dict, str]:
        """提取内容"""
        doc = fitz.open(pdf_path)
        
        pages_text = {}
        text_parts = []
        for i in range(len(doc)):
            page_text = doc[i].get_text()
            if self._should_skip_page(page_text):
                continue
            page_num = i + 1
            pages_text[page_num] = page_text
            truncated = page_text[:2500]
            text_parts.append(f"=== 第{page_num}页 ===\n{truncated}")
        
        doc.close()
        doc_type = self._detect_doc_type(" ".join(text_parts[:3]))
        
        full_text = "\n".join(text_parts)
        if len(full_text) > 18000:
            full_text = full_text[:12000] + "\n...(省略)...\n" + full_text[-4000:]
        
        # 构建prompt - 强调必须包含原文引用
        if doc_type == "GUIDELINE":
            prompt = f"""从以下临床指南中提取推荐意见。

【关键要求】
1. text: 推荐内容的中文总结
2. original_quote: 必须复制原文中的关键句子(英文，20-60字)
3. sources: 引用所在页码，格式["pX"]

文档:
{full_text}

返回JSON:
{{"recommendations": [{{"text": "...", "original_quote": "原文句子", "sources": ["pX"]}}]}}"""
        
        elif doc_type == "REVIEW":
            prompt = f"""从以下综述中提取关键发现。

【关键要求】
1. finding: 发现内容的中文总结
2. original_quote: 必须复制原文中的关键句子(英文，20-60字)
3. sources: 引用所在页码，格式["pX"]

文档:
{full_text}

返回JSON:
{{"key_findings": [{{"finding": "...", "original_quote": "原文句子", "sources": ["pX"]}}]}}"""
        
        else:
            prompt = f"""从以下医学文献中提取关键信息。

【关键要求】
1. finding: 关键发现的中文总结
2. original_quote: 必须复制原文中的关键句子(英文，20-60字)
3. sources: 引用所在页码，格式["pX"]

文档:
{full_text}

返回JSON:
{{"key_findings": [{{"finding": "...", "original_quote": "原文句子", "sources": ["pX"]}}]}}"""
        
        resp, usage = self._call_llm(prompt, 8000, json_mode=True)
        result = self._extract_json(resp)
        
        return result, pages_text, usage, doc_type
    
    def _find_page_for_text(self, text: str, pages_text: Dict[int, str]) -> Optional[int]:
        """精确查找文本所在页码"""
        if not text or len(text) < 10:
            return None
        
        # 清理文本
        search_text = text.lower().strip()
        # 取前40字符搜索
        search_prefix = search_text[:40]
        
        for page_num, page_content in pages_text.items():
            if search_prefix in page_content.lower():
                return page_num
        
        # 如果前缀找不到，尝试关键词匹配
        keywords = re.findall(r'[a-zA-Z]{4,}', text)[:8]
        if len(keywords) < 3:
            return None
        
        best_page = None
        best_score = 0
        for page_num, page_content in pages_text.items():
            page_lower = page_content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in page_lower)
            score = matches / len(keywords)
            if score > best_score and score >= 0.5:
                best_score = score
                best_page = page_num
        
        return best_page
    
    def _verify_and_fix(self, result: Dict, pages_text: Dict[int, str]) -> Tuple[Dict, Dict]:
        """验证并修正source，删除无法验证的条目"""
        stats = {"total": 0, "verified": 0, "fixed": 0, "removed": 0}
        
        # 处理recommendations
        verified_recs = []
        for rec in result.get('recommendations', []):
            stats["total"] += 1
            quote = rec.get('original_quote', '')
            
            # 基于quote查找实际页码
            actual_page = self._find_page_for_text(quote, pages_text)
            
            if actual_page:
                # 获取当前标注的页码
                current_source = rec.get('sources', [])
                current_page = None
                if current_source:
                    match = re.search(r'p(\d+)', str(current_source[0]).lower())
                    if match:
                        current_page = int(match.group(1))
                
                if current_page == actual_page:
                    stats["verified"] += 1
                else:
                    rec['sources'] = [f"p{actual_page}"]
                    stats["fixed"] += 1
                
                verified_recs.append(rec)
            else:
                # 无法验证，删除此条目
                stats["removed"] += 1
        
        result['recommendations'] = verified_recs
        
        # 处理key_findings
        verified_findings = []
        for finding in result.get('key_findings', []):
            stats["total"] += 1
            quote = finding.get('original_quote', '')
            
            actual_page = self._find_page_for_text(quote, pages_text)
            
            if actual_page:
                current_source = finding.get('sources', [])
                current_page = None
                if current_source:
                    match = re.search(r'p(\d+)', str(current_source[0]).lower())
                    if match:
                        current_page = int(match.group(1))
                
                if current_page == actual_page:
                    stats["verified"] += 1
                else:
                    finding['sources'] = [f"p{actual_page}"]
                    stats["fixed"] += 1
                
                verified_findings.append(finding)
            else:
                stats["removed"] += 1
        
        result['key_findings'] = verified_findings
        
        return result, stats
    
    def extract(self, pdf_path: str) -> Dict:
        start_time = time.time()
        
        if self.use_cache and self.cache:
            pdf_hash = self.cache.get_pdf_hash(pdf_path)
            cached = self.cache.get(pdf_hash, self.VERSION)
            if cached:
                cached['from_cache'] = True
                cached['time'] = time.time() - start_time
                return cached
        
        try:
            result, pages_text, usage, doc_type = self._extract_content(pdf_path)
            
            if not result:
                return {"success": False, "error": "提取失败", "version": self.VERSION}
            
            # 强制验证和修正
            result, verify_stats = self._verify_and_fix(result, pages_text)
            
            elapsed = time.time() - start_time
            
            output = {
                "success": True,
                "result": result,
                "doc_type": doc_type,
                "time": elapsed,
                "stats": {
                    "total_pages": len(pages_text),
                    "tokens": usage,
                    "source_verification": verify_stats
                },
                "version": self.VERSION,
                "from_cache": False
            }
            
            if self.use_cache and self.cache:
                self.cache.set(pdf_hash, output, self.VERSION)
            
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e), "version": self.VERSION}
    
    def extract_batch(self, pdf_paths: List[str], max_workers: int = 2) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.extract, pdf_paths))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF文件路径")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    extractor = MedicalPDFExtractorV715(use_cache=not args.no_cache)
    result = extractor.extract(args.pdf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
