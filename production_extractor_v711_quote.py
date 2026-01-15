"""
生产级医学PDF结构化提取器 v7.11 (Quote优化版)
优化点:
1. 调整提示词: 优先引用原文，允许适度概括
2. 增加Few-shot示例引导
3. 增加输入文本长度
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

SKIP_PATTERNS = [
    r'^table\s+of\s+contents?$', r'^contents?$', r'^目录$',
    r'^references?$', r'^bibliography$', r'^参考文献$',
    r'^appendix', r'^附录',
]

def clean_json_string(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text: str) -> Optional[str]:
    text = clean_json_string(text)
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


class MedicalPDFExtractorV711:
    """v7.11 Quote优化版提取器"""
    
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 4500, timeout: int = 300) -> str:
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
        kws = ['recommend', 'should', 'must', 'suggest', 'result', 'finding', 'conclude',
               '建议', '推荐', '应该', '必须', '发现', '结论']
        for kw in kws:
            score += min(low.count(kw), 5) * 0.4
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
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        """优化的提示词: 平衡引用原文和提取完整性"""
        
        # Few-shot示例(简化版)
        fewshot_section = ""
        if fewshot and "expected_output" in fewshot:
            ex = fewshot["expected_output"]
            if doc_type == "GUIDELINE" and "recommendations" in ex:
                sample_recs = ex["recommendations"][:2] if isinstance(ex["recommendations"], list) else []
                if sample_recs:
                    fewshot_section = f"""
示例输出格式:
{{"recommendations": {json.dumps(sample_recs, ensure_ascii=False)[:600]}...}}

"""
        
        if doc_type == "GUIDELINE":
            return f"""你是医学文献信息提取专家。从以下临床指南中提取所有推荐意见。

【提取要求】
1. 尽可能多地提取所有推荐/建议，不要遗漏
2. text字段优先使用原文表述，可适度精简但保持原意
3. 必须标注来源页码sources，格式如["p5"]或["p3-p5"]
4. 提取推荐强度(强推荐/弱推荐)和证据等级(如有)
{fewshot_section}
输出JSON格式:
{{"doc_metadata":{{"title":"","authors":[],"year":"","organization":""}},"scope":{{"target_population":"","clinical_context":""}},"recommendations":[{{"id":"1","text":"推荐内容","strength":"","evidence_level":"","sources":["p页码"]}}],"key_evidence":[{{"evidence":"","sources":["p页码"]}}]}}

文档内容:
{text}

请提取所有推荐意见，返回JSON:"""

        elif doc_type == "REVIEW":
            return f"""你是医学文献信息提取专家。从以下综述/系统评价中提取关键发现。

【提取要求】
1. 尽可能多地提取所有关键发现和结论
2. finding字段优先使用原文表述，可适度精简
3. 必须标注来源页码sources
4. 包含具体数据(如OR、RR、CI等)
{fewshot_section}
输出JSON格式:
{{"doc_metadata":{{"title":"","authors":[],"year":""}},"scope":{{"objectives":"","methods":""}},"key_findings":[{{"id":"F1","finding":"发现内容","sources":["p页码"]}}],"conclusions":{{"main_conclusion":"","implications":"","limitations":""}}}}

文档内容:
{text}

请提取所有关键发现，返回JSON:"""

        else:
            return f"""你是医学文献信息提取专家。从以下医学文献中提取关键信息。

【提取要求】
1. 提取所有关键发现和结论
2. 优先使用原文表述
3. 标注来源页码

输出JSON格式:
{{"doc_metadata":{{"title":"","authors":[],"year":""}},"key_findings":[{{"id":"1","finding":"","sources":["p页码"]}}],"conclusions":{{"main_conclusion":""}}}}

文档内容:
{text}

返回JSON:"""
    
    def _extract_single_pass(self, doc: fitz.Document, doc_type: str, max_pages: int) -> Dict:
        total_pages = len(doc)
        selected = self._select_pages(doc, doc_type, max_pages)
        
        full_text = "\n".join([f"=== p{p['index']+1} ===\n{p['text']}" for p in selected])
        
        # 增加输入文本长度
        max_text_chars = 14000  # 比v7.10增加2000字符
        if len(full_text) > max_text_chars:
            # 保留更多头部内容(通常包含摘要和结论)
            head = int(max_text_chars * 0.6)
            tail = max_text_chars - head
            full_text = full_text[:head] + "\n...[省略]...\n" + full_text[-tail:]
        
        fewshot = self._load_fewshot(doc_type)
        prompt = self._build_prompt(doc_type, full_text, fewshot)
        result = self._call_llm(prompt, max_tokens=4000, timeout=200)
        json_str = extract_first_json(result)
        
        return {
            'data': json.loads(json_str) if json_str else {},
            'stats': {'total_pages': total_pages, 'selected_pages': len(selected), 'strategy': 'quote_v3'}
        }
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            classify_text = "\n".join([doc[i].get_text() for i in range(min(3, total_pages))])[:2000]
            doc_type = self.classify(classify_text)
            
            # 增加选页数量
            max_pages = 45 if total_pages <= MEDIUM_DOC_THRESHOLD else 55
            result = self._extract_single_pass(doc, doc_type, max_pages)
            doc.close()
            
            if result['data']:
                return {"success": True, "doc_type": doc_type, "result": result['data'],
                        "time": time.time() - start_time, "stats": result['stats'], "version": "v7.11_quote"}
            else:
                return {"success": False, "error": "提取结果为空", "time": time.time() - start_time, "version": "v7.11_quote"}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time, "version": "v7.11_quote"}


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    return MedicalPDFExtractorV711().extract(pdf_path)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"类型: {result.get('doc_type')}, 成功: {result.get('success')}, 耗时: {result.get('time', 0):.1f}s")
    if result.get('success'):
        data = result['result']
        items = len(data.get('recommendations', data.get('key_findings', [])))
        print(f"提取条目数: {items}")
