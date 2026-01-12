"""
生产级医学PDF结构化提取器 v7.5
改进：代码层面自适应 - 不同文档大小不同页面处理策略，prompt保持一致
- 短文档(≤20页): 全文提取，不跳过任何页
- 中等文档(21-50页): 全文提取，跳过目录/参考文献/空白页
- 长文档(>50页): 智能选择最多50页关键内容
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

SHORT_DOC_THRESHOLD = 20
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


class AdaptivePageExtractor:
    """自适应页面提取器 - 代码层面控制"""
    
    def __init__(self, max_long_doc_pages: int = 50, min_content_chars: int = 100):
        self.max_long_doc_pages = max_long_doc_pages
        self.min_content_chars = min_content_chars
    
    def _should_skip_page(self, text: str) -> bool:
        """判断是否应跳过此页（目录/参考文献等）"""
        first_lines = '\n'.join(text.lower().strip().split('\n')[:5])
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return True
        # 检测目录页（大量页码格式）
        if len(re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)) > 10:
            return True
        return False
    
    def _is_empty_page(self, text: str) -> bool:
        """判断是否是空白页"""
        return len(re.sub(r'\s+', '', text)) < self.min_content_chars
    
    def _get_page_priority(self, text: str, page_num: int, total_pages: int) -> int:
        """获取页面优先级分数"""
        first_lines = '\n'.join(text.lower().split('\n')[:5])
        priority = 0
        for pattern, score in PRIORITY_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                priority = max(priority, score)
        # 前3页和最后3页加分
        if page_num <= 3 or page_num > total_pages - 3:
            priority += 3
        return priority
    
    def extract_pages(self, doc: fitz.Document) -> Tuple[List[Tuple[int, str]], str]:
        """
        根据文档大小自适应提取页面
        返回: ([(页码, 文本), ...], 文档大小类型)
        """
        total_pages = len(doc)
        
        # 确定文档大小类型
        if total_pages <= SHORT_DOC_THRESHOLD:
            doc_size = 'short'
        elif total_pages <= MEDIUM_DOC_THRESHOLD:
            doc_size = 'medium'
        else:
            doc_size = 'long'
        
        # 短文档：全文提取，不跳过任何有效内容页
        if doc_size == 'short':
            pages = []
            for i in range(total_pages):
                text = doc[i].get_text()
                if not self._is_empty_page(text):  # 只跳过空白页
                    pages.append((i, text))
            return pages, doc_size
        
        # 中等文档：全文提取，跳过目录/参考文献/空白页
        if doc_size == 'medium':
            pages = []
            for i in range(total_pages):
                text = doc[i].get_text()
                if self._is_empty_page(text):
                    continue
                if self._should_skip_page(text):
                    continue
                pages.append((i, text))
            return pages, doc_size
        
        # 长文档：智能选择最多50页
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
        
        # 如果有效页面不超过限制，全返回
        if len(page_info) <= self.max_long_doc_pages:
            return [(p['index'], p['text']) for p in page_info], doc_size
        
        # 智能选择
        selected = set()
        
        # 1. 前5页（元数据、摘要）
        for p in page_info[:5]:
            selected.add(p['index'])
        
        # 2. 最后3页（结论）
        for p in page_info[-3:]:
            selected.add(p['index'])
        
        # 3. 高优先级页面（摘要、结论、推荐等）
        priority_sorted = sorted(page_info, key=lambda x: -x['priority'])
        for p in priority_sorted[:20]:
            if len(selected) >= self.max_long_doc_pages:
                break
            selected.add(p['index'])
        
        # 4. 按内容量填充剩余
        remaining = self.max_long_doc_pages - len(selected)
        if remaining > 0:
            other = [p for p in page_info if p['index'] not in selected]
            other.sort(key=lambda x: -x['char_count'])
            for p in other[:remaining]:
                selected.add(p['index'])
        
        # 按页码顺序返回
        result = [(p['index'], p['text']) for p in page_info if p['index'] in selected]
        result.sort(key=lambda x: x[0])
        return result, doc_size


class MedicalPDFExtractor:
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self.page_extractor = AdaptivePageExtractor()
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 6000) -> str:
        resp = requests.post(
            self.api_url,
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=300
        )
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def _extract_text(self, pdf_path: str) -> Tuple[str, Dict]:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        selected_pages, doc_size = self.page_extractor.extract_pages(doc)
        
        pages_text = []
        for idx, text in selected_pages:
            pages_text.append(f"=== 第{idx+1}页 ===\n{text}")
        
        doc.close()
        
        return "\n".join(pages_text), {
            'total_pages': total_pages,
            'selected_pages': len(selected_pages),
            'doc_size': doc_size
        }
    
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
        prompt = """判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER
- GUIDELINE: 临床指南/诊疗规范/技术评估
- REVIEW: 综述/系统评价/Meta分析
- OTHER: 其他

文档开头：
""" + text[:2500] + """

只返回类型名："""
        result = self._call_llm(prompt, max_tokens=50)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        # 统一的prompt，不根据文档大小变化
        base = f"""医学文献信息提取专家。从{doc_type}文档提取结构化信息。

要求：
1. sources标注页码如["p1"]或["p3-p5"]
2. 只提取原文存在的信息，不编造
3. 提取具体内容，不是章节标题
4. 推荐/建议要完整引用原文

"""
        fewshot_section = ""
        if fewshot:
            ex = fewshot["expected_output"]
            simplified = {"doc_metadata": ex.get("doc_metadata", {})}
            if doc_type == "GUIDELINE" and "recommendations" in ex:
                simplified["recommendations"] = ex["recommendations"][:1]
            fewshot_section = f"示例：\n{json.dumps(simplified, ensure_ascii=False)[:1200]}\n\n"
        
        if doc_type == "GUIDELINE":
            fmt = '{"doc_metadata":{...},"scope":{...},"recommendations":[{"id":"1.1","text":"完整推荐内容","strength":"强度","sources":["px"]}],"key_evidence":[...]}'
        elif doc_type == "REVIEW":
            fmt = '{"doc_metadata":{...},"scope":{...},"key_findings":[{"id":"F1","finding":"具体发现","sources":["px"]}],"conclusions":[...]}'
        else:
            fmt = '{"doc_metadata":{...},"scope":{...},"key_findings":[...],"conclusions":[...]}'
        
        max_len = 12000
        if len(text) > max_len:
            head = text[:max_len // 2]
            tail = text[-(max_len // 2):]
            text = head + "\n\n...[中间部分省略]...\n\n" + tail
        
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
                return {
                    "success": True,
                    "doc_type": doc_type,
                    "result": data,
                    "time": time.time() - start_time,
                    "stats": stats
                }
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
        print(f"文档: {s['doc_size']} ({s['total_pages']}页→{s['selected_pages']}页)")
