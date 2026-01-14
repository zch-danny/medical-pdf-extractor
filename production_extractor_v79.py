"""
生产级医学PDF结构化提取器 v7.9 (32K优化版)
分流架构：
- 短/中文档(≤50页): v7.3智能选择，一次性提取
- 长文档(51-80页): 选60页，一次性提取（利用32K上下文）
- 超长文档(>80页): 先选80页 → MapReduce分段摘要
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

# 阈值配置（32K优化）
MEDIUM_DOC_THRESHOLD = 50   # ≤50页：短/中文档
LONG_DOC_THRESHOLD = 80     # 51-80页：长文档一次性处理
# >80页：超长文档用MapReduce

SKIP_PATTERNS = [
    r'^table\s+of\s+contents?$', r'^contents?$', r'^目录$',
    r'^references?$', r'^bibliography$', r'^参考文献$',
    r'^appendix', r'^附录', r'^acknowledge?ments?$', r'^致谢$',
]

PRIORITY_PATTERNS = [
    (r'abstract|摘要|summary|executive\s+summary', 10),
    (r'conclusion|结论|讨论|discussion', 8),
    (r'recommend|建议|推荐|results?|结果', 7),
    (r'method|方法|材料', 5),
    (r'introduction|背景|background|引言', 4),
]

def clean_json_string(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text: str) -> Optional[str]:
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    text = clean_json_string(text)
    start = text.find('{')
    if start == -1:
        return None
    depth, in_string, escape = 0, False, False
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

class MedicalPDFExtractor:
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 6000, timeout: int = 300) -> str:
        resp = requests.post(self.api_url, json={
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=timeout)
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def _is_empty_page(self, text: str) -> bool:
        return len(re.sub(r'\s+', '', text)) < 100
    
    def _should_skip_page(self, text: str) -> bool:
        first_lines = '\n'.join(text.lower().strip().split('\n')[:5])
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True
        if len(re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)) > 10:
            return True
        return False
    
    def _load_fewshot(self, doc_type: str) -> Optional[Dict]:
        if doc_type not in self._fewshot_cache:
            sample_path = FEWSHOT_DIR / f"{doc_type}_sample.json"
            if sample_path.exists():
                with open(sample_path, encoding='utf-8') as f:
                    self._fewshot_cache[doc_type] = json.load(f)
            else:
                self._fewshot_cache[doc_type] = None
        return self._fewshot_cache[doc_type]
    
    def classify(self, text: str) -> str:
        prompt = "判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER\n- GUIDELINE: 临床指南/诊疗规范/技术评估\n- REVIEW: 综述/系统评价/Meta分析\n- OTHER: 其他\n\n文档开头：\n" + text[:2500] + "\n\n只返回类型名："
        result = self._call_llm(prompt, max_tokens=50, timeout=60)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _score_page(self, text: str, doc_type: str, page_num: int, total_pages: int) -> float:
        """综合评分页面重要性"""
        score = 0.0
        low = text.lower()
        first5 = '\n'.join(low.split('\n')[:5])
        
        # 标题信号
        title_patterns = [r'\babstract\b', r'\bintroduction\b', r'\bmethods?\b', r'\bresults?\b',
                         r'\bdiscussion\b', r'\bconclusions?\b', r'\bsummary\b', r'\brecommend',
                         r'摘要', r'引言', r'方法', r'结果', r'讨论', r'结论', r'建议', r'推荐']
        if any(re.search(p, first5) for p in title_patterns):
            score += 4.0
        
        # 关键词信号
        kw_gdl = ['recommend', 'should', 'must', 'class i', 'class ii', 'level a', 'level b', '建议', '必须']
        kw_rev = ['result', 'finding', 'significant', 'p<', ' ci ', ' or ', ' rr ', '显著']
        kws = kw_gdl if doc_type == 'GUIDELINE' else kw_rev
        for kw in kws:
            score += min(low.count(kw), 3) * 0.5
        
        # 密度信号
        chars = len(text)
        score += min(chars / 2000, 3.0)
        
        # 位置加分
        if page_num <= 5 or page_num > total_pages - 3:
            score += 2.0
        
        return score
    
    def _select_pages(self, doc: fitz.Document, doc_type: str, max_pages: int) -> List[Dict]:
        """智能选择页面"""
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
        
        # 强制保留前5后3
        keep = set(range(min(5, total_pages)))
        keep.update(range(max(0, total_pages - 3), total_pages))
        
        # 按分数选剩余
        for p in sorted(page_infos, key=lambda x: -x['score']):
            if len(keep) >= max_pages:
                break
            keep.add(p['index'])
        
        return sorted([p for p in page_infos if p['index'] in keep], key=lambda x: x['index'])
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        base = f"医学文献信息提取专家。从{doc_type}文档提取结构化信息。\n\n要求：\n1. sources标注页码如[\"p1\"]或[\"p3-p5\"]\n2. 只提取原文存在的信息，不编造\n3. 推荐/建议要完整引用原文\n\n"
        
        fewshot_section = ""
        if fewshot:
            ex = fewshot["expected_output"]
            simplified = {"doc_metadata": ex.get("doc_metadata", {})}
            if doc_type == "GUIDELINE" and "recommendations" in ex:
                simplified["recommendations"] = ex["recommendations"][:1]
            fewshot_section = f"示例：\n{json.dumps(simplified, ensure_ascii=False)[:1200]}\n\n"
        
        fmt_map = {
            "GUIDELINE": '{"doc_metadata":{...},"scope":{...},"recommendations":[{"id":"1.1","text":"完整推荐内容","strength":"强度","sources":["px"]}],"key_evidence":[...]}',
            "REVIEW": '{"doc_metadata":{...},"scope":{...},"key_findings":[{"id":"F1","finding":"具体发现","sources":["px"]}],"conclusions":[...]}',
            "OTHER": '{"doc_metadata":{...},"scope":{...},"key_findings":[...],"conclusions":[...]}'
        }
        
        return f"{base}{fewshot_section}格式：{fmt_map.get(doc_type, fmt_map['OTHER'])}\n\n文档：\n{text}\n\n返回JSON："
    
    # ========== 短/中/长文档：一次性提取 ==========
    def _extract_single_pass(self, doc: fitz.Document, doc_type: str, max_pages: int, strategy: str) -> Dict:
        """一次性提取（短/中/长文档通用）"""
        total_pages = len(doc)
        selected = self._select_pages(doc, doc_type, max_pages)
        
        # 32K上下文可以放更多内容
        full_text = "\n".join([f"=== 第{p['index']+1}页 ===\n{p['text']}" for p in selected])
        
        # 32K上下文：输入可以到20K字符，输出留8K tokens
        if len(full_text) > 20000:
            full_text = full_text[:10000] + "\n\n...[中间省略]...\n\n" + full_text[-10000:]
        
        fewshot = self._load_fewshot(doc_type)
        prompt = self._build_prompt(doc_type, full_text, fewshot)
        result = self._call_llm(prompt, max_tokens=8000, timeout=180)
        json_str = extract_first_json(result)
        
        return {
            'data': json.loads(json_str) if json_str else {},
            'stats': {'total_pages': total_pages, 'selected_pages': len(selected), 'strategy': strategy}
        }
    
    # ========== 超长文档：MapReduce ==========
    def _extract_mapreduce(self, doc: fitz.Document, doc_type: str) -> Dict:
        """超长文档(>80页)：先选80页 → MapReduce"""
        total_pages = len(doc)
        selected = self._select_pages(doc, doc_type, max_pages=80)
        
        if not selected:
            return {'data': {}, 'stats': {'total_pages': total_pages, 'selected_pages': 0, 'chunks': 0, 'strategy': 'mapreduce'}}
        
        # 分块（32K下每块可以更大：25页）
        chunk_size = 25
        overlap = 3
        chunks = []
        i = 0
        while i < len(selected):
            j = min(i + chunk_size, len(selected))
            chunk_pages = selected[i:j]
            chunk_text = "\n".join([f"=== 第{p['index']+1}页 ===\n{p['text']}" for p in chunk_pages])
            chunks.append({
                'text': chunk_text[:12000],  # 32K下可以放更多
                'page_range': f"p{chunk_pages[0]['index']+1}-p{chunk_pages[-1]['index']+1}"
            })
            i = j - overlap if j - overlap > i else j
        
        # MAP阶段
        chunk_results = []
        def extract_chunk(chunk_info: Dict) -> Dict:
            chunk_text, page_range = chunk_info['text'], chunk_info['page_range']
            if doc_type == "GUIDELINE":
                prompt = f"""从以下医学指南片段({page_range})提取信息，仅返回JSON：

{chunk_text}

提取：
1. doc_metadata: title, authors, year (如有)
2. recommendations: [{{"id":"编号","text":"完整推荐内容","strength":"推荐强度","sources":["页码"]}}]
3. key_evidence: [{{"evidence":"证据内容","sources":["页码"]}}]
"""
            else:
                prompt = f"""从以下医学文献片段({page_range})提取信息，仅返回JSON：

{chunk_text}

提取：
1. doc_metadata: title, authors, year (如有)
2. key_findings: [{{"id":"编号","finding":"发现内容","sources":["页码"]}}]
3. conclusions: {{"main_conclusion":"...","implications":"..."}} (如有)
"""
            try:
                result = self._call_llm(prompt, max_tokens=4000, timeout=150)
                json_str = extract_first_json(result)
                return json.loads(json_str) if json_str else {}
            except:
                return {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(extract_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                r = future.result()
                if r:
                    chunk_results.append(r)
        
        merged = self._merge_results(chunk_results, doc_type)
        return {
            'data': merged,
            'stats': {'total_pages': total_pages, 'selected_pages': len(selected), 'chunks': len(chunks), 'strategy': 'mapreduce'}
        }
    
    def _merge_results(self, results: List[Dict], doc_type: str) -> Dict:
        merged = {'doc_metadata': {}, 'scope': {}}
        
        for r in results:
            if 'doc_metadata' in r and isinstance(r['doc_metadata'], dict):
                for k, v in r['doc_metadata'].items():
                    if k not in merged['doc_metadata'] or not merged['doc_metadata'][k]:
                        merged['doc_metadata'][k] = v
        
        if doc_type == 'GUIDELINE':
            all_recs, seen = [], set()
            for r in results:
                recs = r.get('recommendations', [])
                if isinstance(recs, list):
                    for rec in recs:
                        text = (rec.get('text') or '')[:120]
                        if text and text not in seen:
                            seen.add(text)
                            all_recs.append(rec)
            merged['recommendations'] = all_recs[:25]
            all_evi = []
            for r in results:
                ev = r.get('key_evidence', [])
                if isinstance(ev, list):
                    all_evi.extend(ev)
            merged['key_evidence'] = all_evi[:20]
        else:
            all_find, seen = [], set()
            for r in results:
                ff = r.get('key_findings', [])
                if isinstance(ff, list):
                    for f in ff:
                        text = (f.get('finding') or '')[:120]
                        if text and text not in seen:
                            seen.add(text)
                            all_find.append(f)
            merged['key_findings'] = all_find[:20]
            for r in results:
                if r.get('conclusions'):
                    merged['conclusions'] = r['conclusions']
                    break
        
        return merged
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            classify_text = "\n".join([doc[i].get_text() for i in range(min(3, total_pages))])[:2500]
            doc_type = self.classify(classify_text)
            
            # 根据页数选择策略
            if total_pages <= MEDIUM_DOC_THRESHOLD:
                # 短/中文档：一次性提取，最多选50页
                result = self._extract_single_pass(doc, doc_type, max_pages=50, strategy='single_short')
            elif total_pages <= LONG_DOC_THRESHOLD:
                # 长文档(51-80页)：一次性提取，最多选60页（利用32K）
                result = self._extract_single_pass(doc, doc_type, max_pages=60, strategy='single_long')
            else:
                # 超长文档(>80页)：MapReduce
                result = self._extract_mapreduce(doc, doc_type)
            
            doc.close()
            
            if result['data']:
                return {"success": True, "doc_type": doc_type, "result": result['data'], "time": time.time() - start_time, "stats": result['stats']}
            else:
                return {"success": False, "error": "提取结果为空", "time": time.time() - start_time}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}

def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    return MedicalPDFExtractor().extract(pdf_path)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"类型: {result.get('doc_type')}, 成功: {result.get('success')}, 耗时: {result.get('time', 0):.1f}s")
    if 'stats' in result:
        s = result['stats']
        print(f"策略: {s.get('strategy')} | 页面: {s.get('total_pages')}→{s.get('selected_pages')}")
