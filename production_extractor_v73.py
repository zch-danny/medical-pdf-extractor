"""
生产级医学PDF结构化提取器 v7.3
智能页面选择版：跳过目录/参考文献，最多选择50页关键内容
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

class SmartPageSelector:
    """智能页面选择器：跳过无用页面，选择关键内容"""
    
    def __init__(self, max_pages: int = 50, min_content_chars: int = 100):
        self.max_pages = max_pages
        self.min_content_chars = min_content_chars
    
    def _is_empty_page(self, text: str) -> bool:
        return len(re.sub(r'\s+', '', text)) < self.min_content_chars
    
    def _should_skip_page(self, text: str) -> bool:
        first_lines = '\n'.join(text.lower().strip().split('\n')[:5])
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True
        # 检测目录页（多个页码引用）
        if len(re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)) > 10:
            return True
        return False
    
    def _get_page_priority(self, text: str, page_num: int, total_pages: int) -> int:
        first_lines = '\n'.join(text.lower().split('\n')[:5])
        priority = 0
        for pattern, score in PRIORITY_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                priority = max(priority, score)
        # 首尾页面加分
        if page_num <= 3 or page_num > total_pages - 3:
            priority += 3
        return priority
    
    def select_pages(self, doc: fitz.Document) -> Tuple[List[Tuple[int, str]], Dict]:
        total_pages = len(doc)
        
        # 收集所有有效页面
        page_info = []
        for i in range(total_pages):
            text = doc[i].get_text()
            if self._is_empty_page(text):
                continue
            if self._should_skip_page(text):
                continue
            priority = self._get_page_priority(text, i + 1, total_pages)
            page_info.append({
                'index': i, 
                'text': text, 
                'priority': priority,
                'char_count': len(text.strip())
            })
        
        # 如果页面不多，全部保留
        if len(page_info) <= self.max_pages:
            return [(p['index'], p['text']) for p in page_info], {
                'total_pages': total_pages,
                'selected_pages': len(page_info),
                'strategy': 'all'
            }
        
        # 智能选择最多max_pages页
        selected = set()
        
        # 1. 首5页必选
        for p in page_info[:5]:
            selected.add(p['index'])
        
        # 2. 末3页必选
        for p in page_info[-3:]:
            selected.add(p['index'])
        
        # 3. 按优先级选择高分页
        sorted_by_priority = sorted(page_info, key=lambda x: -x['priority'])
        for p in sorted_by_priority:
            if len(selected) >= self.max_pages:
                break
            selected.add(p['index'])
        
        # 4. 如果还不够，按内容长度补充
        if len(selected) < self.max_pages:
            remaining = self.max_pages - len(selected)
            unselected = [p for p in page_info if p['index'] not in selected]
            sorted_by_length = sorted(unselected, key=lambda x: -x['char_count'])
            for p in sorted_by_length[:remaining]:
                selected.add(p['index'])
        
        # 按页码顺序输出
        result = [(p['index'], p['text']) for p in page_info if p['index'] in selected]
        result.sort(key=lambda x: x[0])
        
        return result, {
            'total_pages': total_pages,
            'selected_pages': len(result),
            'strategy': 'smart_select'
        }

class MedicalPDFExtractor:
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self.page_selector = SmartPageSelector()
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 6000) -> str:
        resp = requests.post(self.api_url, json={
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=300)
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def _extract_text(self, pdf_path: str) -> Tuple[str, Dict]:
        doc = fitz.open(pdf_path)
        selected_pages, stats = self.page_selector.select_pages(doc)
        pages_text = [f"=== 第{idx+1}页 ===\n{text}" for idx, text in selected_pages]
        doc.close()
        return "\n".join(pages_text), stats
    
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
        result = self._call_llm(prompt, max_tokens=50)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        base = f"医学文献信息提取专家。从{doc_type}文档提取结构化信息。\n\n要求：\n1. sources标注页码如[\"p1\"]或[\"p3-p5\"]\n2. 只提取原文存在的信息，不编造\n3. 提取具体内容，不是章节标题\n4. 推荐/建议要完整引用原文\n\n"
        
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
            "OTHER": '{"doc_metadata":{...},"scope":{...},"key_findings":[...],conclusions":[...]}'
        }
        fmt = fmt_map.get(doc_type, fmt_map["OTHER"])
        
        # 长文本截断
        if len(text) > 12000:
            text = text[:6000] + "\n\n...[中间部分省略]...\n\n" + text[-6000:]
        
        return f"{base}{fewshot_section}格式：{fmt}\n\n文档：\n{text}\n\n返回JSON："

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            text, stats = self._extract_text(pdf_path)
            if len(text.strip()) < 100:
                return {"success": False, "error": "PDF内容过少", "time": time.time() - start_time}
            
            doc_type = self.classify(text)
            fewshot = self._load_fewshot(doc_type)
            prompt = self._build_prompt(doc_type, text, fewshot)
            result = self._call_llm(prompt)
            
            json_str = extract_first_json(result)
            if json_str:
                data = json.loads(json_str)
                return {"success": True, "doc_type": doc_type, "result": data, "time": time.time() - start_time, "stats": stats}
            else:
                return {"success": False, "error": "未找到有效JSON", "time": time.time() - start_time}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON解析失败: {e}", "time": time.time() - start_time}
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
        print(f"页面: {s['total_pages']}→{s['selected_pages']} ({s['strategy']})")
