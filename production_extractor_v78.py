"""
生产级医学PDF结构化提取器 v7.8
优化版：改进长文档处理策略
- 短文档(≤15页): 全部保留
- 中等文档(16-50页): 跳过目录/参考文献
- 长文档(>50页): 两阶段提取(摘要+细节)
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

SHORT_DOC_THRESHOLD = 15
MEDIUM_DOC_THRESHOLD = 50

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

class SmartExtractor:
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
    
    def _get_page_priority(self, text: str, page_num: int, total_pages: int) -> int:
        first_lines = '\n'.join(text.lower().split('\n')[:5])
        priority = 0
        for pattern, score in PRIORITY_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                priority = max(priority, score)
        if page_num <= 3 or page_num > total_pages - 3:
            priority += 3
        return priority
    
    def _select_pages(self, doc: fitz.Document) -> Tuple[List[Tuple[int, str, int]], str, Dict]:
        total_pages = len(doc)
        
        if total_pages <= SHORT_DOC_THRESHOLD:
            doc_size = 'short'
        elif total_pages <= MEDIUM_DOC_THRESHOLD:
            doc_size = 'medium'
        else:
            doc_size = 'long'
        
        pages = []
        for i in range(total_pages):
            text = doc[i].get_text()
            if self._is_empty_page(text):
                continue
            if doc_size != 'short' and self._should_skip_page(text):
                continue
            priority = self._get_page_priority(text, i + 1, total_pages)
            pages.append((i, text, priority))
        
        # 长文档：选择最重要的40页
        if doc_size == 'long' and len(pages) > 40:
            selected_indices = set()
            # 前5页后3页必选
            for p in pages[:5]:
                selected_indices.add(p[0])
            for p in pages[-3:]:
                selected_indices.add(p[0])
            # 按优先级选20页
            sorted_by_priority = sorted(pages, key=lambda x: -x[2])
            for p in sorted_by_priority:
                if len(selected_indices) >= 40:
                    break
                selected_indices.add(p[0])
            pages = [p for p in pages if p[0] in selected_indices]
            pages.sort(key=lambda x: x[0])
        
        stats = {'total_pages': total_pages, 'selected_pages': len(pages), 'doc_size': doc_size}
        return pages, doc_size, stats
    
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
        prompt = "判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER\n- GUIDELINE: 临床指南/诊疗规范\n- REVIEW: 综述/系统评价/Meta分析\n- OTHER: 其他\n\n文档开头：\n" + text[:2500] + "\n\n只返回类型名："
        result = self._call_llm(prompt, max_tokens=50, timeout=60)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _extract_long_doc(self, pages: List[Tuple[int, str, int]], doc_type: str, fewshot: Optional[Dict]) -> Dict:
        """长文档两阶段提取"""
        # 阶段1: 从前10页提取元数据和范围
        front_text = "\n".join([f"=== 第{p[0]+1}页 ===\n{p[1]}" for p in pages[:10]])[:8000]
        
        meta_prompt = f"""从医学{doc_type}文档提取基本信息，返回JSON：

{front_text}

格式：{{"doc_metadata":{{"title":"标题","authors":["作者"],"journal":"期刊","year":"年份"}},"scope":{{"objective":"目的","population":"人群","focus_areas":["领域"]}}}}"""
        
        meta_result = self._call_llm(meta_prompt, max_tokens=2000, timeout=120)
        meta_json = extract_first_json(meta_result)
        base_data = json.loads(meta_json) if meta_json else {}
        
        # 阶段2: 从高优先级页面提取核心内容
        high_priority = sorted(pages, key=lambda x: -x[2])[:25]
        high_priority.sort(key=lambda x: x[0])
        core_text = "\n".join([f"=== 第{p[0]+1}页 ===\n{p[1]}" for p in high_priority])
        
        if len(core_text) > 14000:
            core_text = core_text[:7000] + "\n...[省略]...\n" + core_text[-7000:]
        
        if doc_type == "GUIDELINE":
            content_prompt = f"""从指南文档提取推荐建议和关键证据：

{core_text}

要求：
1. 提取所有明确的推荐建议，包含完整推荐文本
2. 标注推荐强度(Class I/II/III, Level A/B/C等)
3. sources标注页码

返回JSON：{{"recommendations":[{{"id":"1.1","text":"完整推荐内容","strength":"强度","sources":["px"]}}],"key_evidence":[{{"evidence":"证据","sources":["px"]}}]}}"""
        else:
            content_prompt = f"""从{doc_type}文档提取主要发现和结论：

{core_text}

要求：
1. 提取具体的研究发现，包含数据
2. sources标注页码

返回JSON：{{"key_findings":[{{"id":"F1","finding":"发现","sources":["px"]}}],"conclusions":{{"main":"主要结论","implications":"临床意义"}}}}"""
        
        content_result = self._call_llm(content_prompt, max_tokens=4000, timeout=180)
        content_json = extract_first_json(content_result)
        content_data = json.loads(content_json) if content_json else {}
        
        return {**base_data, **content_data}
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        base = f"医学文献信息提取专家。从{doc_type}文档提取结构化信息。\n\n要求：\n1. sources标注页码\n2. 只提取原文存在的信息\n3. 推荐/建议要完整引用\n\n"
        
        fewshot_section = ""
        if fewshot:
            ex = fewshot["expected_output"]
            simplified = {"doc_metadata": ex.get("doc_metadata", {})}
            if doc_type == "GUIDELINE" and "recommendations" in ex:
                simplified["recommendations"] = ex["recommendations"][:1]
            fewshot_section = f"示例：\n{json.dumps(simplified, ensure_ascii=False)[:1200]}\n\n"
        
        fmt_map = {
            "GUIDELINE": '{"doc_metadata":{...},"scope":{...},"recommendations":[{"id":"1.1","text":"推荐内容","strength":"强度","sources":["px"]}],"key_evidence":[...]}',
            "REVIEW": '{"doc_metadata":{...},"scope":{...},"key_findings":[{"id":"F1","finding":"发现","sources":["px"]}],"conclusions":{...}}',
            "OTHER": '{"doc_metadata":{...},"scope":{...},"key_findings":[...],"conclusions":{...}}'
        }
        
        return f"{base}{fewshot_section}格式：{fmt_map.get(doc_type, fmt_map['OTHER'])}\n\n文档：\n{text}\n\n返回JSON："
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages, doc_size, stats = self._select_pages(doc)
            doc.close()
            
            if not pages:
                return {"success": False, "error": "PDF内容过少", "time": time.time() - start_time, "stats": stats}
            
            classify_text = "\n".join([p[1] for p in pages[:3]])[:2500]
            doc_type = self.classify(classify_text)
            fewshot = self._load_fewshot(doc_type)
            
            # 长文档使用两阶段提取
            if doc_size == 'long':
                data = self._extract_long_doc(pages, doc_type, fewshot)
            else:
                full_text = "\n".join([f"=== 第{p[0]+1}页 ===\n{p[1]}" for p in pages])
                if len(full_text) > 12000:
                    full_text = full_text[:6000] + "\n...[省略]...\n" + full_text[-6000:]
                prompt = self._build_prompt(doc_type, full_text, fewshot)
                result = self._call_llm(prompt)
                json_str = extract_first_json(result)
                data = json.loads(json_str) if json_str else {}
            
            if data:
                return {"success": True, "doc_type": doc_type, "result": data, "time": time.time() - start_time, "stats": stats}
            else:
                return {"success": False, "error": "提取结果为空", "time": time.time() - start_time, "stats": stats}
        
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}

def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    return SmartExtractor().extract(pdf_path)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"类型: {result.get('doc_type')}, 成功: {result.get('success')}, 耗时: {result.get('time', 0):.1f}s")
    if 'stats' in result:
        s = result['stats']
        print(f"文档: {s['doc_size']} ({s['total_pages']}页→{s['selected_pages']}页)")
