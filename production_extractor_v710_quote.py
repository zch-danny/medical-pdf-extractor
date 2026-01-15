"""
生产级医学PDF结构化提取器 v7.10 (Quote-then-Structure优化版) - 修复版
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

MEDIUM_DOC_THRESHOLD = 50
LONG_DOC_THRESHOLD = 80

SKIP_PATTERNS = [
    r'^table\s+of\s+contents?$', r'^contents?$', r'^目录$',
    r'^references?$', r'^bibliography$', r'^参考文献$',
    r'^appendix', r'^附录',
]

def clean_json_string(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text: str) -> Optional[str]:
    """改进的JSON提取：支持从文本中间找JSON"""
    text = clean_json_string(text)
    # 尝试找markdown代码块
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        text = match.group(1)
    
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


class MedicalPDFExtractorV710:
    """Quote-then-Structure优化版提取器 - 修复版"""
    
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 4000, timeout: int = 300) -> str:
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
        return False
    
    def classify(self, text: str) -> str:
        prompt = "判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER\n文档开头：\n" + text[:2000] + "\n只返回类型名："
        result = self._call_llm(prompt, max_tokens=50, timeout=60)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _score_page(self, text: str, doc_type: str, page_num: int, total_pages: int) -> float:
        score = 0.0
        low = text.lower()
        first5 = '\n'.join(low.split('\n')[:5])
        
        title_patterns = [r'\babstract\b', r'\bintroduction\b', r'\bresults?\b',
                         r'\bconclusions?\b', r'\brecommend', r'摘要', r'结论', r'推荐']
        if any(re.search(p, first5) for p in title_patterns):
            score += 4.0
        
        kws = ['recommend', 'should', 'result', 'finding', '建议', '发现']
        for kw in kws:
            score += min(low.count(kw), 3) * 0.5
        
        score += min(len(text) / 2000, 3.0)
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
    
    def _build_quote_prompt(self, doc_type: str, text: str) -> str:
        """简化的Quote-then-Structure提示词：强调必须引用原文"""
        
        if doc_type == "GUIDELINE":
            return f"""从以下临床指南中提取信息，返回JSON。

【重要要求】
1. recommendations中的text字段必须是原文的直接引用，不要改写或概括
2. 每条推荐必须标注来源页码sources
3. 只提取文档中明确存在的推荐，不要编造

输出格式：
{{"doc_metadata":{{"title":"","authors":[],"year":"","organization":""}},"recommendations":[{{"id":"1","text":"原文直接引用","strength":"","sources":["p页码"]}}],"key_evidence":[{{"evidence":"","sources":["p页码"]}}]}}

文档：
{text}

返回JSON："""

        elif doc_type == "REVIEW":
            return f"""从以下综述/系统评价中提取信息，返回JSON。

【重要要求】
1. key_findings中的finding字段必须是原文的直接引用，不要改写
2. 每条发现必须标注来源页码sources
3. 只提取文档中明确存在的发现，不要编造

输出格式：
{{"doc_metadata":{{"title":"","authors":[],"year":""}},"key_findings":[{{"id":"F1","finding":"原文直接引用","sources":["p页码"]}}],"conclusions":{{"main_conclusion":"","implications":""}}}}

文档：
{text}

返回JSON："""

        else:
            return f"""从以下医学文献中提取信息，返回JSON。

【重要要求】
1. key_findings中的内容必须是原文的直接引用
2. 标注来源页码sources
3. 不要编造内容

输出格式：
{{"doc_metadata":{{"title":"","authors":[],"year":""}},"key_findings":[{{"id":"1","finding":"原文直接引用","sources":["p页码"]}}],"conclusions":{{"main_conclusion":""}}}}

文档：
{text}

返回JSON："""
    
    def _extract_single_pass(self, doc: fitz.Document, doc_type: str, max_pages: int) -> Dict:
        total_pages = len(doc)
        selected = self._select_pages(doc, doc_type, max_pages)
        
        full_text = "\n".join([f"=== p{p['index']+1} ===\n{p['text']}" for p in selected])
        
        # 更保守的token估算
        max_text_chars = 12000  # 约9000 tokens，留足够输出空间
        if len(full_text) > max_text_chars:
            half = max_text_chars // 2
            full_text = full_text[:half] + "\n...[省略]...\n" + full_text[-half:]
        
        prompt = self._build_quote_prompt(doc_type, full_text)
        result = self._call_llm(prompt, max_tokens=3000, timeout=180)
        json_str = extract_first_json(result)
        
        return {
            'data': json.loads(json_str) if json_str else {},
            'stats': {'total_pages': total_pages, 'selected_pages': len(selected), 'strategy': 'quote_v2'}
        }
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            classify_text = "\n".join([doc[i].get_text() for i in range(min(3, total_pages))])[:2000]
            doc_type = self.classify(classify_text)
            
            max_pages = 40 if total_pages <= MEDIUM_DOC_THRESHOLD else 50
            result = self._extract_single_pass(doc, doc_type, max_pages)
            doc.close()
            
            if result['data']:
                return {
                    "success": True, "doc_type": doc_type, "result": result['data'],
                    "time": time.time() - start_time, "stats": result['stats'], "version": "v7.10_quote"
                }
            else:
                return {"success": False, "error": "提取结果为空", "time": time.time() - start_time, "version": "v7.10_quote"}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time, "version": "v7.10_quote"}


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    return MedicalPDFExtractorV710().extract(pdf_path)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"类型: {result.get('doc_type')}, 成功: {result.get('success')}, 耗时: {result.get('time', 0):.1f}s")
