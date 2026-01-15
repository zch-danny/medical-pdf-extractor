#!/usr/bin/env python3
"""
医学PDF结构化提取器 v7.16
优化改进:
1. 更完整的文档覆盖 - 不截断关键内容
2. 更精确的source验证 - 使用n-gram匹配
3. 保留更多条目 - 降低删除阈值
4. 改进JSON解析鲁棒性
"""

import os, sys, json, re, time, hashlib, sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz
import requests

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


class MedicalPDFExtractorV716:
    VERSION = "v7.16"
    
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
        
        resp = requests.post(self.api_url, json=payload, timeout=300)
        result = resp.json()
        return result["choices"][0]["message"]["content"], result.get("usage", {})
    
    def _extract_json(self, text: str) -> Dict:
        """鲁棒的JSON解析"""
        # 直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 修复截断JSON
        for key in ['recommendations', 'key_findings']:
            if f'"{key}"' in text:
                start = text.find('{')
                # 找最后完整对象
                last_complete = text.rfind('}]')
                if last_complete > 0:
                    try:
                        return json.loads(text[start:last_complete+2] + '}')
                    except:
                        pass
                # 补全括号
                last_brace = text.rfind('}')
                if last_brace > start:
                    substr = text[start:last_brace+1]
                    suffix = ']' * (substr.count('[') - substr.count(']'))
                    suffix += '}' * (substr.count('{') - substr.count('}'))
                    try:
                        return json.loads(substr + suffix)
                    except:
                        pass
        
        # 正则回退
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {}
    
    def _should_skip_page(self, text: str) -> bool:
        first_line = text.strip().split('\n')[0].lower() if text.strip() else ""
        return any(re.match(p, first_line, re.I) for p in self.SKIP_PATTERNS)
    
    def _detect_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['guideline', 'recommendation', '指南', '推荐', 'statement']):
            return "GUIDELINE"
        if any(kw in text_lower for kw in ['systematic review', 'meta-analysis', '系统综述']):
            return "REVIEW"
        return "OTHER"
    
    def _get_ngrams(self, text: str, n: int = 3) -> set:
        """获取n-gram集合用于匹配"""
        text = text.lower().strip()
        words = re.findall(r'[a-zA-Z]+|\d+', text)
        if len(words) < n:
            return set(words)
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    def _find_page_for_text(self, text: str, pages_text: Dict[int, str]) -> Tuple[Optional[int], float]:
        """查找文本所在页码，返回(页码, 置信度)"""
        if not text or len(text) < 15:
            return None, 0.0
        
        # 方法1: 精确前缀匹配
        search_prefix = text[:50].lower().strip()
        for page_num, page_content in pages_text.items():
            if search_prefix in page_content.lower():
                return page_num, 1.0
        
        # 方法2: n-gram匹配
        text_ngrams = self._get_ngrams(text)
        if not text_ngrams:
            return None, 0.0
        
        best_page = None
        best_score = 0.0
        
        for page_num, page_content in pages_text.items():
            page_ngrams = self._get_ngrams(page_content)
            if page_ngrams:
                overlap = len(text_ngrams & page_ngrams)
                score = overlap / len(text_ngrams)
                if score > best_score:
                    best_score = score
                    best_page = page_num
        
        # 降低阈值到0.3以保留更多条目
        return (best_page, best_score) if best_score >= 0.3 else (None, 0.0)
    
    def _extract_content(self, pdf_path: str) -> Tuple[Dict, Dict, Dict, str]:
        """提取内容 - 改进版"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        pages_text = {}
        text_parts = []
        
        for i in range(total_pages):
            page_text = doc[i].get_text()
            if self._should_skip_page(page_text):
                continue
            page_num = i + 1
            pages_text[page_num] = page_text
            # 增加每页字符限制以获取更完整内容
            truncated = page_text[:3500]
            text_parts.append(f"=== 第{page_num}页 ===\n{truncated}")
        
        doc.close()
        doc_type = self._detect_doc_type(" ".join(text_parts[:3]))
        
        # 智能截断：保留更多内容
        full_text = "\n".join(text_parts)
        if len(full_text) > 25000:
            # 保留前15000和后8000
            full_text = full_text[:15000] + "\n...(中间省略)...\n" + full_text[-8000:]
        
        # 改进的提取Prompt - 强调完整性和准确性
        if doc_type == "GUIDELINE":
            prompt = f"""你是医学文献信息提取专家。请从以下临床指南中提取所有推荐意见。

【任务要求】
1. 提取文档中的所有推荐/建议，不要遗漏
2. text: 用中文总结推荐内容
3. original_quote: 复制原文关键句(英文，30-80字)
4. sources: 标注引用所在页码，格式["pX"]
   - 查看"=== 第X页 ==="标记确定页码
   - 必须准确，不要猜测

【文档内容】
{full_text}

【输出格式】
返回JSON:
{{"recommendations": [
  {{"text": "推荐内容中文总结", "original_quote": "原文关键句", "sources": ["pX"]}}
]}}

请确保提取完整，输出JSON:"""
        
        elif doc_type == "REVIEW":
            prompt = f"""你是医学文献信息提取专家。请从以下综述中提取所有关键发现。

【任务要求】
1. 提取文档中的所有关键发现和结论
2. finding: 用中文总结发现
3. original_quote: 复制原文关键句(英文，30-80字)
4. sources: 标注引用所在页码，格式["pX"]

【文档内容】
{full_text}

【输出格式】
返回JSON:
{{"key_findings": [
  {{"finding": "关键发现中文总结", "original_quote": "原文关键句", "sources": ["pX"]}}
]}}

请确保提取完整，输出JSON:"""
        
        else:
            prompt = f"""你是医学文献信息提取专家。请从以下文献中提取所有关键信息。

【任务要求】
1. 提取文档中的关键发现、结论和重要信息
2. finding: 用中文总结
3. original_quote: 复制原文关键句(英文，30-80字)
4. sources: 标注引用所在页码，格式["pX"]

【文档内容】
{full_text}

【输出格式】
返回JSON:
{{"key_findings": [
  {{"finding": "关键信息中文总结", "original_quote": "原文关键句", "sources": ["pX"]}}
]}}

请确保提取完整，输出JSON:"""
        
        # 增加max_tokens以避免截断
        resp, usage = self._call_llm(prompt, 10000, json_mode=True)
        result = self._extract_json(resp)
        
        return result, pages_text, usage, doc_type
    
    def _verify_and_fix(self, result: Dict, pages_text: Dict[int, str]) -> Tuple[Dict, Dict]:
        """验证并修正source - 改进版"""
        stats = {"total": 0, "verified": 0, "fixed": 0, "removed": 0}
        
        def process_items(items: List[Dict], key_field: str) -> List[Dict]:
            verified = []
            for item in items:
                stats["total"] += 1
                quote = item.get('original_quote', '')
                text_content = item.get(key_field, '')
                
                # 优先用quote查找，其次用text
                search_text = quote if quote else text_content
                actual_page, confidence = self._find_page_for_text(search_text, pages_text)
                
                if actual_page:
                    current_sources = item.get('sources', [])
                    current_page = None
                    if current_sources:
                        match = re.search(r'p(\d+)', str(current_sources[0]).lower())
                        if match:
                            current_page = int(match.group(1))
                    
                    if current_page == actual_page:
                        stats["verified"] += 1
                    else:
                        item['sources'] = [f"p{actual_page}"]
                        stats["fixed"] += 1
                    
                    verified.append(item)
                else:
                    # 如果text_content本身有效，尝试用它查找
                    if quote and text_content:
                        alt_page, alt_conf = self._find_page_for_text(text_content, pages_text)
                        if alt_page:
                            item['sources'] = [f"p{alt_page}"]
                            stats["fixed"] += 1
                            verified.append(item)
                            continue
                    stats["removed"] += 1
            
            return verified
        
        if 'recommendations' in result:
            result['recommendations'] = process_items(result.get('recommendations', []), 'text')
        
        if 'key_findings' in result:
            result['key_findings'] = process_items(result.get('key_findings', []), 'finding')
        
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    extractor = MedicalPDFExtractorV716(use_cache=not args.no_cache)
    result = extractor.extract(args.pdf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
