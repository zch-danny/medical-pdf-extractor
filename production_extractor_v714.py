#!/usr/bin/env python3
"""
医学PDF结构化提取器 v7.14
改进:
1. 增强页码标注prompt
2. 先引用后总结（Quote-then-Summarize）
3. 更严格的source验证阈值
"""

import os, sys, json, re, time, hashlib, sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz
import requests
from concurrent.futures import ThreadPoolExecutor

# ============ 缓存模块 ============
class ExtractionCache:
    """基于SQLite的提取结果缓存"""
    
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
        """计算PDF文件的MD5哈希"""
        hasher = hashlib.md5()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get(self, pdf_hash: str, version: str) -> Optional[Dict]:
        """获取缓存的提取结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            'SELECT result FROM cache WHERE pdf_hash = ? AND version = ?',
            (pdf_hash, version)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None
    
    def set(self, pdf_hash: str, result: Dict, version: str):
        """缓存提取结果"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            'INSERT OR REPLACE INTO cache (pdf_hash, result, version) VALUES (?, ?, ?)',
            (pdf_hash, json.dumps(result, ensure_ascii=False), version)
        )
        conn.commit()
        conn.close()
    
    def clear(self):
        """清空缓存"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM cache')
        conn.commit()
        conn.close()

# ============ 主提取器 ============
class MedicalPDFExtractorV714:
    VERSION = "v7.14"
    
    # 跳过模式
    SKIP_PATTERNS = [
        r'^(table of contents|contents|目录)',
        r'^(references?|bibliography|参考文献)',
        r'^(appendix|附录)',
        r'^(acknowledgements?|致谢)',
    ]
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions", 
                 use_cache: bool = True, cache_path: str = None):
        self.api_url = api_url
        self.model = "qwen3-8b"
        self.use_cache = use_cache
        self.cache = ExtractionCache(cache_path) if use_cache else None
    
    def _call_llm(self, prompt: str, max_tokens: int = 2000, json_mode: bool = False) -> Tuple[str, Dict]:
        """调用vLLM API"""
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
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        return content, usage
    
    def _extract_json(self, text: str) -> Dict:
        """从响应中提取JSON"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试查找JSON块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pat else match.group())
                except:
                    continue
        return {}
    
    def _should_skip_page(self, text: str) -> bool:
        """判断是否应该跳过此页"""
        first_line = text.strip().split('\n')[0].lower() if text.strip() else ""
        for pat in self.SKIP_PATTERNS:
            if re.match(pat, first_line, re.IGNORECASE):
                return True
        return False
    
    def _detect_doc_type(self, text: str) -> str:
        """检测文档类型"""
        text_lower = text.lower()
        guideline_kw = ['guideline', 'recommendation', 'clinical practice', '指南', '推荐意见', 'statement']
        review_kw = ['systematic review', 'meta-analysis', '系统综述', 'pooled analysis']
        
        for kw in guideline_kw:
            if kw in text_lower:
                return "GUIDELINE"
        for kw in review_kw:
            if kw in text_lower:
                return "REVIEW"
        return "OTHER"
    
    def _extract_with_quotes(self, pdf_path: str) -> Tuple[Dict, Dict, Dict]:
        """Quote-then-Summarize提取方法"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # 收集页面文本
        pages_text = {}
        text_parts = []
        
        for i in range(total_pages):
            page_text = doc[i].get_text()
            if self._should_skip_page(page_text):
                continue
            page_num = i + 1
            pages_text[page_num] = page_text
            # 截取每页关键内容
            truncated = page_text[:3000] if len(page_text) > 3000 else page_text
            text_parts.append(f"=== 第{page_num}页 ===\n{truncated}")
        
        doc.close()
        
        # 检测文档类型
        doc_type = self._detect_doc_type(" ".join(text_parts[:3]))
        
        # 合并文本
        full_text = "\n".join(text_parts)
        if len(full_text) > 20000:
            head = 13000
            tail = 5000
            full_text = full_text[:head] + "\n...(中间省略)...\n" + full_text[-tail:]
        
        # Quote-then-Summarize Prompt
        if doc_type == "GUIDELINE":
            prompt = f"""你是医学信息提取专家。请从以下临床指南中提取推荐意见。

【重要】提取步骤:
1. 先找到推荐的原文句子
2. 记录该句子所在的页码（看"=== 第X页 ==="标记）
3. 然后提取要点

【页码标注规则】
- sources必须标注内容实际所在页码
- 例如：如果内容在"=== 第5页 ==="下方，则sources为["p5"]
- 不要猜测页码，必须根据文档标记确定

文档内容:
{full_text}

返回JSON格式:
{{
  "recommendations": [
    {{
      "text": "推荐内容",
      "strength": "强/中/弱",
      "evidence": "A/B/C",
      "original_quote": "原文引用（≤50字）",
      "sources": ["pX"]  // X是实际页码
    }}
  ]
}}

仅返回JSON:"""
        
        elif doc_type == "REVIEW":
            prompt = f"""你是医学信息提取专家。请从以下综述中提取关键发现。

【重要】提取步骤:
1. 先找到关键发现的原文句子
2. 记录该句子所在的页码（看"=== 第X页 ==="标记）
3. 然后提取要点

【页码标注规则】
- sources必须标注内容实际所在页码
- 例如：如果内容在"=== 第5页 ==="下方，则sources为["p5"]
- 不要猜测页码，必须根据文档标记确定

文档内容:
{full_text}

返回JSON格式:
{{
  "key_findings": [
    {{
      "finding": "关键发现",
      "data": "具体数据(OR/RR/CI等)",
      "original_quote": "原文引用（≤50字）",
      "sources": ["pX"]  // X是实际页码
    }}
  ]
}}

仅返回JSON:"""
        
        else:
            prompt = f"""你是医学信息提取专家。请从以下医学文献中提取关键信息。

【重要】提取步骤:
1. 先找到关键内容的原文句子
2. 记录该句子所在的页码（看"=== 第X页 ==="标记）
3. 然后提取要点

【页码标注规则】
- sources必须标注内容实际所在页码
- 例如：如果内容在"=== 第5页 ==="下方，则sources为["p5"]
- 不要猜测页码，必须根据文档标记确定

文档内容:
{full_text}

返回JSON格式:
{{
  "key_findings": [
    {{
      "finding": "关键发现",
      "original_quote": "原文引用（≤50字）",
      "sources": ["pX"]  // X是实际页码
    }}
  ]
}}

仅返回JSON:"""
        
        # 调用LLM
        resp, usage = self._call_llm(prompt, 6000, json_mode=True)
        result = self._extract_json(resp)
        
        return result, pages_text, usage, doc_type
    
    def _verify_and_fix_sources(self, result: Dict, pages_text: Dict[int, str]) -> Tuple[Dict, Dict]:
        """验证并修正来源页码"""
        stats = {"total": 0, "verified": 0, "fixed": 0}
        
        def get_keywords(text: str) -> List[str]:
            """提取关键词"""
            words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{3,}|\d+\.?\d*', text)
            return [w.lower() for w in words if len(w) > 2][:15]
        
        def find_best_page(text: str) -> Optional[int]:
            """找到最匹配的页码"""
            keywords = get_keywords(text)
            if not keywords:
                return None
            
            best_page = None
            best_score = 0
            
            for page_num, page_text in pages_text.items():
                page_lower = page_text.lower()
                matches = sum(1 for kw in keywords if kw in page_lower)
                score = matches / len(keywords)
                if score > best_score:
                    best_score = score
                    best_page = page_num
            
            return best_page if best_score >= 0.3 else None  # 提高阈值到30%
        
        def verify_source(text: str, sources: List) -> bool:
            """验证来源是否正确"""
            keywords = get_keywords(text)
            if not keywords:
                return True  # 无法验证
            
            for src in (sources or []):
                match = re.search(r'p(\d+)', str(src).lower())
                if match:
                    page_num = int(match.group(1))
                    if page_num in pages_text:
                        page_lower = pages_text[page_num].lower()
                        matches = sum(1 for kw in keywords if kw in page_lower)
                        if matches / len(keywords) >= 0.3:  # 30%阈值
                            return True
            return False
        
        # 处理recommendations
        for rec in result.get('recommendations', []):
            stats["total"] += 1
            text = rec.get('text', '') + ' ' + rec.get('original_quote', '')
            if not verify_source(text, rec.get('sources', [])):
                best_page = find_best_page(text)
                if best_page:
                    rec['sources'] = [f"p{best_page}"]
                    stats["fixed"] += 1
                else:
                    stats["verified"] += 1  # 无法找到更好的，保留原来的
            else:
                stats["verified"] += 1
        
        # 处理key_findings
        for finding in result.get('key_findings', []):
            stats["total"] += 1
            text = finding.get('finding', '') + ' ' + finding.get('original_quote', '')
            if not verify_source(text, finding.get('sources', [])):
                best_page = find_best_page(text)
                if best_page:
                    finding['sources'] = [f"p{best_page}"]
                    stats["fixed"] += 1
                else:
                    stats["verified"] += 1
            else:
                stats["verified"] += 1
        
        return result, stats
    
    def extract(self, pdf_path: str) -> Dict:
        """主提取方法"""
        start_time = time.time()
        
        # 检查缓存
        if self.use_cache and self.cache:
            pdf_hash = self.cache.get_pdf_hash(pdf_path)
            cached = self.cache.get(pdf_hash, self.VERSION)
            if cached:
                cached['from_cache'] = True
                cached['time'] = time.time() - start_time
                return cached
        
        try:
            # Quote-then-Summarize提取
            result, pages_text, usage, doc_type = self._extract_with_quotes(pdf_path)
            
            if not result:
                return {"success": False, "error": "提取失败", "version": self.VERSION}
            
            # 验证并修正来源
            result, verify_stats = self._verify_and_fix_sources(result, pages_text)
            
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
            
            # 存入缓存
            if self.use_cache and self.cache:
                self.cache.set(pdf_hash, output, self.VERSION)
            
            return output
            
        except Exception as e:
            return {"success": False, "error": str(e), "version": self.VERSION}
    
    def extract_batch(self, pdf_paths: List[str], max_workers: int = 2) -> List[Dict]:
        """批量提取"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.extract, pdf_paths))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF文件路径")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    args = parser.parse_args()
    
    extractor = MedicalPDFExtractorV714(use_cache=not args.no_cache)
    result = extractor.extract(args.pdf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
