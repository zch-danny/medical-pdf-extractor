#!/usr/bin/env python3
"""
医学PDF结构化提取器 v7.20
目标：准确度、精确度、完整性全方位提升（允许适度增加耗时）
核心策略：分页筛选 + 章节分段抽取 + 严格校验 + 去重合并
"""

import os
import json
import re
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

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


class MedicalPDFExtractorV720:
    VERSION = "v7.20"

    SKIP_PATTERNS = [
        r'^(table of contents|contents|目录)',
        r'^(references?|bibliography|参考文献)',
        r'^(appendix|附录)',
    ]

    MEDIUM_DOC_THRESHOLD = 50
    LONG_DOC_THRESHOLD = 80
    SELECT_PAGES_MEDIUM = 60
    SELECT_PAGES_LONG = 80

    PER_PAGE_CHARS = 2500
    SECTION_MAX_CHARS = 16000
    NGRAM_THRESHOLD = 0.5
    MODEL_MAX_CONTEXT = 16384
    CHARS_PER_TOKEN = 1.2  # 保守估算，英文为主

    SECTION_RULES = {
        "GUIDELINE": [
            ("recommendations", [r'\brecommend', r'\bguideline', r'\bstatement', r'\bclass\s*i', r'\blevel\s*[abc]', r'建议', r'推荐', r'声明', r'指南']),
            ("evidence", [r'\bevidence', r'\bresults?', r'\btrial', r'\bmeta-?analysis', r'证据', r'结果', r'研究', r'试验']),
            ("conclusion", [r'\bconclusion', r'\bsummary', r'\bdiscussion', r'结论', r'总结', r'讨论']),
        ],
        "REVIEW": [
            ("results", [r'\bresults?', r'\bfindings?', r'\bmeta-?analysis', r'\bstatistical', r'结果', r'发现', r'显著', r'统计']),
            ("discussion", [r'\bdiscussion', r'\bimplication', r'讨论', r'意义', r'局限']),
            ("conclusion", [r'\bconclusion', r'\bsummary', r'结论', r'总结']),
        ],
        "OTHER": [
            ("key", [r'\bresults?', r'\bconclusion', r'\bsummary', r'结果', r'结论', r'总结']),
        ]
    }

    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions",
                 use_cache: bool = True, cache_path: str = None):
        self.api_url = api_url
        self.model = "qwen3-8b"
        self.use_cache = use_cache
        self.cache = ExtractionCache(cache_path) if use_cache else None

    def _call_llm(self, prompt: str, max_tokens: int = 8000, json_mode: bool = False, retries: int = 3, auto_limit: bool = True) -> Tuple[str, Dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # 动态限制 max_tokens 防止超出上下文
        if auto_limit:
            est_input_tokens = int(len(prompt) / self.CHARS_PER_TOKEN)
            available = self.MODEL_MAX_CONTEXT - est_input_tokens - 100
            if available < max_tokens:
                max_tokens = max(1000, available)
                payload["max_tokens"] = max_tokens

        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.post(self.api_url, json=payload, timeout=300)
                data = resp.json()
                if "choices" not in data:
                    err_msg = data.get("error", {}).get("message", str(data))
                    raise RuntimeError(f"LLM响应无choices: {err_msg}")
                return data["choices"][0]["message"]["content"], data.get("usage", {})
            except Exception as e:
                last_error = e
                import time as _t
                _t.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM调用失败(重试{retries}次): {last_error}")

    def _extract_json(self, text: str) -> Dict:
        """鲁棒JSON解析"""
        try:
            return json.loads(text)
        except Exception:
            pass

        # 修复截断
        for key in ['recommendations', 'key_findings']:
            if f'"{key}"' in text:
                start = text.find('{')
                last_complete = text.rfind('}]')
                if last_complete > 0:
                    try:
                        return json.loads(text[start:last_complete + 2] + '}')
                    except Exception:
                        pass
        # 兜底：尝试找完整JSON块
        match = re.search(r'\{.*\}', text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

    def _is_empty_page(self, text: str) -> bool:
        return len(re.sub(r'\s+', '', text)) < 100

    def _should_skip_page(self, text: str) -> bool:
        first_lines = '\n'.join(text.lower().strip().split('\n')[:5])
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True
        # 目录/索引式页
        if len(re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)) > 10:
            return True
        return False

    def _detect_doc_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['guideline', 'recommendation', '指南', '推荐', 'statement']):
            return "GUIDELINE"
        if any(kw in text_lower for kw in ['systematic review', 'meta-analysis', '系统综述', 'meta analysis']):
            return "REVIEW"
        return "OTHER"

    def _score_page(self, text: str, doc_type: str, page_num: int, total_pages: int) -> float:
        score = 0.0
        low = text.lower()
        first5 = '\n'.join(low.split('\n')[:5])

        title_patterns = [r'\babstract\b', r'\bintroduction\b', r'\bmethods?\b', r'\bresults?\b',
                          r'\bdiscussion\b', r'\bconclusions?\b', r'\bsummary\b', r'\brecommend',
                          r'摘要', r'引言', r'方法', r'结果', r'讨论', r'结论', r'建议', r'推荐']
        if any(re.search(p, first5) for p in title_patterns):
            score += 4.0

        kw_gdl = ['recommend', 'should', 'must', 'class i', 'class ii', 'level a', 'level b', '建议', '必须']
        kw_rev = ['result', 'finding', 'significant', 'p<', ' ci ', ' or ', ' rr ', '显著']
        kws = kw_gdl if doc_type == 'GUIDELINE' else kw_rev
        for kw in kws:
            score += min(low.count(kw), 3) * 0.5

        chars = len(text)
        score += min(chars / 2000, 3.0)

        if page_num <= 5 or page_num > total_pages - 3:
            score += 2.0

        return score

    def _select_pages(self, doc: fitz.Document, doc_type: str, max_pages: int) -> List[Dict]:
        total_pages = len(doc)
        page_infos = []

        for i in range(total_pages):
            txt = doc[i].get_text()
            if self._is_empty_page(txt) or self._should_skip_page(txt):
                continue
            score = self._score_page(txt, doc_type, i + 1, total_pages)
            page_infos.append({'index': i, 'text': txt, 'score': score})

        if len(page_infos) <= max_pages:
            return page_infos

        keep = set(range(min(5, total_pages)))
        keep.update(range(max(0, total_pages - 3), total_pages))

        for p in sorted(page_infos, key=lambda x: -x['score']):
            if len(keep) >= max_pages:
                break
            keep.add(p['index'])

        return sorted([p for p in page_infos if p['index'] in keep], key=lambda x: x['index'])

    def _assign_section(self, text: str, doc_type: str) -> str:
        rules = self.SECTION_RULES.get(doc_type, [])
        first_lines = '\n'.join(text.lower().strip().split('\n')[:5])
        for section, patterns in rules:
            for p in patterns:
                if re.search(p, first_lines):
                    return section
        return "body"

    def _build_text_from_pages(self, pages: List[Dict], max_chars: int) -> str:
        parts = []
        total = 0
        for p in pages:
            page_num = p['index'] + 1
            truncated = p['text'][:self.PER_PAGE_CHARS]
            block = f"=== 第{page_num}页 ===\n{truncated}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    def _chunk_pages(self, pages: List[Dict], chunk_size: int = 25, overlap: int = 3) -> List[List[Dict]]:
        chunks = []
        i = 0
        while i < len(pages):
            j = min(i + chunk_size, len(pages))
            chunks.append(pages[i:j])
            i = j - overlap if j - overlap > i else j
        return chunks

    def _build_prompt(self, doc_type: str, section: str, full_text: str) -> str:
        if doc_type == "GUIDELINE":
            task = "提取该章节中的临床推荐/建议"
            fmt = '{"recommendations": [{"text": "中文推荐总结", "original_quote": "原文关键句", "sources": ["pX"]}]}'
            item_key = "recommendations"
            content_field = "text"
        else:
            task = "提取该章节中的关键发现/结论"
            fmt = '{"key_findings": [{"finding": "中文发现总结", "original_quote": "原文关键句", "sources": ["pX"]}]}'
            item_key = "key_findings"
            content_field = "finding"

        prompt = f"""你是医学文献信息提取专家。
当前章节: {section}
任务: {task}

要求：
1. 只基于文档内容，不要编造
2. 每条必须包含 original_quote（原文连续句子，英文30-80字）
3. sources 标注页码，格式 [\"pX\"]，严格依据 \"=== 第X页 ===\" 标记
4. 如果无法找到原文句子或页码，请不要输出该条
5. 尽量覆盖该章节的所有要点，不遗漏

文档内容：
{full_text}

输出JSON格式：
{fmt}

请只输出JSON。"""
        return prompt

    def _merge_items(self, items: List[Dict], key_field: str) -> List[Dict]:
        seen = set()
        seen_quote = set()
        merged = []
        for it in items:
            text = (it.get(key_field) or "").strip()
            quote = (it.get('original_quote') or "").strip()
            if not text:
                continue
            norm = re.sub(r'\W+', '', text.lower())
            if norm in seen:
                continue
            if quote and quote.lower() in seen_quote:
                continue
            seen.add(norm)
            if quote:
                seen_quote.add(quote.lower())
            merged.append(it)
        return merged

    def _get_ngrams(self, text: str, n: int = 3) -> set:
        """生成n-gram，支持中英文混合"""
        text = text.lower().strip()
        ngrams = set()
        
        # 英文：按单词分
        en_words = re.findall(r'[a-zA-Z]+|\d+', text)
        if len(en_words) >= n:
            ngrams.update(' '.join(en_words[i:i+n]) for i in range(len(en_words)-n+1))
        elif en_words:
            ngrams.update(en_words)
        
        # 中文：按字符分（每n个字符为一个单元）
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if len(cn_chars) >= n:
            ngrams.update(''.join(cn_chars[i:i+n]) for i in range(len(cn_chars)-n+1))
        elif cn_chars:
            ngrams.add(''.join(cn_chars))
        
        return ngrams

    def _find_page_for_text(self, text: str, pages_text: Dict[int, str]) -> Tuple[Optional[int], float]:
        """查找文本所在页码，支持中英文"""
        if not text or len(text) < 10:
            return None, 0.0

        text_lower = text.lower().strip()
        
        # 方法1：精确子串匹配（前60字符）
        search_prefix = text_lower[:60]
        for page_num, page_content in pages_text.items():
            page_lower = page_content[:self.PER_PAGE_CHARS].lower()
            if search_prefix in page_lower:
                return page_num, 1.0
        
        # 方法2：中文短语子串匹配（提取中文部分）
        cn_phrase = ''.join(re.findall(r'[\u4e00-\u9fff]', text_lower))
        if len(cn_phrase) >= 8:
            search_cn = cn_phrase[:20]
            for page_num, page_content in pages_text.items():
                page_cn = ''.join(re.findall(r'[\u4e00-\u9fff]', page_content[:self.PER_PAGE_CHARS].lower()))
                if search_cn in page_cn:
                    return page_num, 0.95

        # 方法3：n-gram模糊匹配（中英文混合）
        text_ngrams = self._get_ngrams(text)
        if not text_ngrams:
            return None, 0.0

        best_page = None
        best_score = 0.0
        for page_num, page_content in pages_text.items():
            page_ngrams = self._get_ngrams(page_content[:self.PER_PAGE_CHARS])
            if page_ngrams:
                overlap = len(text_ngrams & page_ngrams)
                score = overlap / len(text_ngrams)
                if score > best_score:
                    best_score = score
                    best_page = page_num

        return (best_page, best_score) if best_score >= self.NGRAM_THRESHOLD else (None, 0.0)

    def _verify_and_fix(self, result: Dict, pages_text: Dict[int, str]) -> Tuple[Dict, Dict]:
        stats = {"total": 0, "verified": 0, "fixed": 0, "removed": 0}

        def process(items: List[Dict], key_field: str) -> List[Dict]:
            verified = []
            for item in items:
                stats["total"] += 1
                quote = item.get('original_quote', '')
                text_content = item.get(key_field, '')
                search_text = quote if quote else text_content

                actual_page, conf = self._find_page_for_text(search_text, pages_text)
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
                    stats["removed"] += 1
            return verified

        if 'recommendations' in result:
            result['recommendations'] = process(result.get('recommendations', []), 'text')
        if 'key_findings' in result:
            result['key_findings'] = process(result.get('key_findings', []), 'finding')

        return result, stats

    def _extract_sections(self, pages: List[Dict], doc_type: str) -> Tuple[Dict, Dict]:
        section_buckets: Dict[str, List[Dict]] = {}
        for p in pages:
            sec = self._assign_section(p['text'], doc_type)
            section_buckets.setdefault(sec, []).append(p)

        all_items = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for section, sec_pages in section_buckets.items():
            if not sec_pages:
                continue

            # 过长则分块
            if len(sec_pages) > 30:
                chunks = self._chunk_pages(sec_pages, chunk_size=20, overlap=3)
            else:
                chunks = [sec_pages]

            for chunk in chunks:
                full_text = self._build_text_from_pages(chunk, self.SECTION_MAX_CHARS)
                if not full_text.strip():
                    continue
                prompt = self._build_prompt(doc_type, section, full_text)
                resp, usage = self._call_llm(prompt, max_tokens=8000, json_mode=True)
                data = self._extract_json(resp)

                usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
                usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
                usage_total["total_tokens"] += usage.get("total_tokens", 0)

                if doc_type == "GUIDELINE":
                    items = data.get('recommendations', [])
                else:
                    items = data.get('key_findings', [])

                all_items.extend(items)

        if doc_type == "GUIDELINE":
            merged = self._merge_items(all_items, 'text')
            return {"recommendations": merged}, usage_total
        else:
            merged = self._merge_items(all_items, 'finding')
            return {"key_findings": merged}, usage_total

    def _extract_content(self, pdf_path: str) -> Tuple[Dict, Dict, Dict, str, int]:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # 先用前几页判断类型
        preview_text = []
        pages_text: Dict[int, str] = {}
        for i in range(total_pages):
            page_text = doc[i].get_text()
            if self._should_skip_page(page_text):
                continue
            pages_text[i + 1] = page_text
            if len(preview_text) < 3:
                preview_text.append(page_text[:1000])
        doc_type = self._detect_doc_type(" ".join(preview_text))

        # 选择页面
        if total_pages <= self.MEDIUM_DOC_THRESHOLD:
            selected = [{"index": i, "text": pages_text[i + 1]} for i in range(total_pages) if (i + 1) in pages_text]
        elif total_pages <= self.LONG_DOC_THRESHOLD:
            selected = self._select_pages(doc, doc_type, self.SELECT_PAGES_MEDIUM)
        else:
            selected = self._select_pages(doc, doc_type, self.SELECT_PAGES_LONG)
        doc.close()

        if not selected:
            return {}, pages_text, {}, doc_type, 0

        result, usage = self._extract_sections(selected, doc_type)
        return result, pages_text, usage, doc_type, len(selected)

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
            result, pages_text, usage, doc_type, selected_count = self._extract_content(pdf_path)
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
                    "selected_pages": selected_count,
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

    extractor = MedicalPDFExtractorV720(use_cache=not args.no_cache)
    result = extractor.extract(args.pdf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
