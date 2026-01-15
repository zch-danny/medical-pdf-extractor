#!/usr/bin/env python3
"""
生产级医学PDF结构化提取器 v7.13
架构优化:
1. 分层提取 - 阶段1扫描定位，阶段2精细提取
2. 结构化约束输出 - vLLM json_object模式
3. 增量缓存 - PDF hash缓存
4. 批处理支持 - 多PDF并行
"""

import os, sys, json, re, time, hashlib, sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
import requests

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


class MedicalPDFExtractorV713:
    """v7.13 分层提取 + 结构化输出 + 缓存 + 批处理"""
    
    VERSION = "v7.13"
    
    # JSON Schema定义
    GUIDELINE_SCHEMA = {
        "type": "object",
        "properties": {
            "doc_metadata": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": "string"},
                    "organization": {"type": "string"}
                }
            },
            "key_sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "type": {"type": "string"},
                        "summary": {"type": "string"}
                    }
                }
            },
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "strength": {"type": "string"},
                        "evidence_level": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["text", "sources"]
                }
            },
            "key_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "evidence": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        },
        "required": ["recommendations"]
    }
    
    REVIEW_SCHEMA = {
        "type": "object",
        "properties": {
            "doc_metadata": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": "string"}
                }
            },
            "key_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "finding": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["finding", "sources"]
                }
            },
            "conclusions": {
                "type": "object",
                "properties": {
                    "main_conclusion": {"type": "string"},
                    "implications": {"type": "string"},
                    "limitations": {"type": "string"}
                }
            }
        },
        "required": ["key_findings"]
    }
    
    SKIP_PATTERNS = [
        r'^table\s+of\s+contents?$', r'^contents?$', r'^references?$',
        r'^bibliography$', r'^acknowledgments?$', r'^appendix',
        r'^参考文献$', r'^目录$', r'^附录',
    ]
    
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions",
                 use_cache: bool = True, cache_path: str = None):
        self.api_url = api_url
        self.model = "qwen3-8b"
        self.cache = ExtractionCache(cache_path) if use_cache else None
    
    def _call_llm(self, prompt: str, max_tokens: int = 4000, 
                  json_mode: bool = False) -> Tuple[str, Dict]:
        """调用LLM，支持JSON模式"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            resp = requests.post(self.api_url, json=payload, timeout=300)
            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            usage = result.get("usage", {})
            return content.strip(), usage
        except Exception as e:
            return "", {}
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """从文本中提取JSON"""
        # 直接尝试解析(json_mode下应该是纯JSON)
        try:
            return json.loads(text)
        except:
            pass
        
        # 尝试```json块
        match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # 尝试找{...}
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None
    
    def _classify(self, text: str) -> str:
        """分类文档类型"""
        prompt = f"""分析医学文献类型，只返回一个词: GUIDELINE 或 REVIEW 或 OTHER

文本:
{text[:2000]}

类型:"""
        resp, _ = self._call_llm(prompt, 20)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in resp.upper():
                return t
        return "OTHER"
    
    # ===== 阶段1: 快速扫描 =====
    def _stage1_scan(self, doc: fitz.Document, doc_type: str) -> List[Dict]:
        """阶段1: 快速扫描定位关键段落"""
        total = len(doc)
        
        # 构建页面摘要
        page_summaries = []
        for i in range(total):
            text = doc[i].get_text()
            if len(text.strip()) < 100:
                continue
            
            first_line = text.strip().split('\n')[0].lower()
            if any(re.match(p, first_line) for p in self.SKIP_PATTERNS):
                continue
            
            # 取前500字作为摘要
            summary = text[:500].replace('\n', ' ')
            page_summaries.append(f"p{i+1}: {summary}")
        
        # 限制总长度
        summaries_text = "\n".join(page_summaries)[:8000]
        
        # LLM快速扫描
        if doc_type == "GUIDELINE":
            scan_prompt = f"""快速扫描以下文献各页摘要，识别包含推荐意见的页面。

{summaries_text}

返回JSON，列出包含推荐内容的页码:
{{"key_pages": [页码数字列表], "doc_title": "文档标题"}}"""
        else:
            scan_prompt = f"""快速扫描以下文献各页摘要，识别包含关键发现的页面。

{summaries_text}

返回JSON，列出包含关键发现的页码:
{{"key_pages": [页码数字列表], "doc_title": "文档标题"}}"""
        
        resp, _ = self._call_llm(scan_prompt, 200, json_mode=True)
        scan_result = self._extract_json(resp)
        
        if not scan_result:
            # 回退：选择前10页和最后3页
            key_pages = list(range(1, min(11, total+1))) + list(range(max(1, total-2), total+1))
            key_pages = sorted(set(key_pages))
        else:
            key_pages = scan_result.get("key_pages", [])
            # 确保包含前3页和最后2页
            key_pages = sorted(set(key_pages + [1, 2, 3] + ([total-1, total] if total > 3 else [])))
        
        # 过滤有效页码
        key_pages = [p for p in key_pages if 1 <= p <= total][:30]
        
        return key_pages
    
    # ===== 阶段2: 精细提取 =====
    def _stage2_extract(self, doc: fitz.Document, doc_type: str, 
                        key_pages: List[int]) -> Dict:
        """阶段2: 对关键页面精细提取"""
        # 构建精选文本
        pages_text = {}
        text_parts = []
        for p in key_pages:
            page_text = doc[p-1].get_text()
            pages_text[p] = page_text
            text_parts.append(f"=== p{p} ===\n{page_text}")
        
        full_text = "\n".join(text_parts)
        
        # 截断
        max_chars = 14000
        if len(full_text) > max_chars:
            head = int(max_chars * 0.65)
            tail = max_chars - head - 50
            full_text = full_text[:head] + "\n...(省略)...\n" + full_text[-tail:]
        
        # 构建提取prompt
        if doc_type == "GUIDELINE":
            extract_prompt = f"""从以下临床指南中提取所有推荐意见。

要求:
1. 提取所有推荐/建议，使用原文表述
2. sources标注实际页码，如["p5"]
3. 提取推荐强度和证据等级

文档内容:
{full_text}

返回JSON格式(必须包含recommendations数组):"""
        
        elif doc_type == "REVIEW":
            extract_prompt = f"""从以下综述中提取关键发现。

要求:
1. 提取所有关键发现和结论
2. sources标注实际页码
3. 包含具体数据(OR/RR/CI等)

文档内容:
{full_text}

返回JSON格式(必须包含key_findings数组):"""
        
        else:
            extract_prompt = f"""从以下医学文献中提取关键信息。

要求:
1. 提取关键发现和结论
2. 标注来源页码

文档内容:
{full_text}

返回JSON格式(包含key_findings数组):"""
        
        # 使用JSON模式提取
        resp, usage = self._call_llm(extract_prompt, 5000, json_mode=True)
        result = self._extract_json(resp)
        
        return result, pages_text, usage
    
    def _fix_sources(self, result: Dict, pages_text: Dict[int, str]) -> Dict:
        """后处理修正页码"""
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
        
        def verify_source(text: str, sources: List) -> bool:
            for s in (sources or []):
                m = re.search(r'p(\d+)', str(s), re.I)
                if m:
                    page_num = int(m.group(1))
                    if page_num in pages_text:
                        words = [w for w in norm(text).split() if len(w) > 3][:10]
                        if len(words) < 2:
                            return True
                        match = sum(1 for w in words if w in norm(pages_text[page_num]))
                        if match >= max(2, len(words) * 0.25):
                            return True
            return False
        
        for rec in result.get('recommendations', []):
            text = rec.get('text', '')
            if text and not verify_source(text, rec.get('sources', [])):
                best = find_best_page(text)
                if best:
                    rec['sources'] = [f"p{best}"]
        
        for finding in result.get('key_findings', []):
            text = finding.get('finding', '')
            if text and not verify_source(text, finding.get('sources', [])):
                best = find_best_page(text)
                if best:
                    finding['sources'] = [f"p{best}"]
        
        return result
    
    def extract(self, pdf_path: str, use_cache: bool = True) -> Dict:
        """主提取函数"""
        start_time = time.time()
        
        # 检查缓存
        pdf_hash = None
        if self.cache and use_cache:
            pdf_hash = self.cache.get_pdf_hash(pdf_path)
            cached = self.cache.get(pdf_hash, self.VERSION)
            if cached:
                cached['from_cache'] = True
                cached['time'] = 0.01
                return cached
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages < 2:
                return {"success": False, "error": "PDF页数太少", "version": self.VERSION}
            
            # 分类
            first_text = "\n".join(doc[i].get_text()[:1000] for i in range(min(3, total_pages)))
            doc_type = self._classify(first_text)
            
            # 阶段1: 快速扫描
            key_pages = self._stage1_scan(doc, doc_type)
            
            # 阶段2: 精细提取
            result, pages_text, usage = self._stage2_extract(doc, doc_type, key_pages)
            
            doc.close()
            
            if result:
                # 后处理修正页码
                result = self._fix_sources(result, pages_text)
                
                output = {
                    "success": True,
                    "result": result,
                    "doc_type": doc_type,
                    "time": time.time() - start_time,
                    "stats": {
                        "total_pages": total_pages,
                        "key_pages": key_pages,
                        "tokens": usage
                    },
                    "version": self.VERSION,
                    "from_cache": False
                }
                
                # 保存缓存
                if self.cache and pdf_hash:
                    self.cache.set(pdf_hash, output, self.VERSION)
                
                return output
            else:
                return {"success": False, "error": "提取失败", "version": self.VERSION}
                
        except Exception as e:
            return {"success": False, "error": str(e), "version": self.VERSION}
    
    def extract_batch(self, pdf_paths: List[str], max_workers: int = 3,
                      use_cache: bool = True) -> List[Dict]:
        """批量提取多个PDF"""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(self.extract, pdf, use_cache): pdf 
                for pdf in pdf_paths
            }
            
            for future in as_completed(future_to_pdf):
                pdf = future_to_pdf[future]
                try:
                    result = future.result()
                    result['pdf'] = pdf
                    results.append(result)
                except Exception as e:
                    results.append({
                        'pdf': pdf,
                        'success': False,
                        'error': str(e),
                        'version': self.VERSION
                    })
        
        # 按原始顺序排序
        pdf_order = {pdf: i for i, pdf in enumerate(pdf_paths)}
        results.sort(key=lambda x: pdf_order.get(x.get('pdf', ''), 999))
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  单个PDF: python production_extractor_v713.py <pdf_path>")
        print("  批量处理: python production_extractor_v713.py --batch <pdf1> <pdf2> ...")
        print("  清除缓存: python production_extractor_v713.py --clear-cache")
        sys.exit(1)
    
    extractor = MedicalPDFExtractorV713()
    
    if sys.argv[1] == "--clear-cache":
        extractor.cache.clear()
        print("缓存已清除")
    elif sys.argv[1] == "--batch":
        results = extractor.extract_batch(sys.argv[2:])
        for r in results:
            print(f"{r.get('pdf', '?').split('/')[-1]}: {'成功' if r.get('success') else '失败'} "
                  f"{'(缓存)' if r.get('from_cache') else ''} {r.get('time', 0):.1f}s")
    else:
        result = extractor.extract(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
