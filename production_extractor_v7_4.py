"""
生产级医学PDF结构化提取器 v7.4
改进：自适应策略 - 根据文档大小选择不同处理方式
- 短文档(≤20页): 全文提取，详细prompt
- 中等文档(21-50页): 全文提取，精简prompt
- 长文档(>50页): 智能页面选择，摘要式prompt
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

# 文档大小阈值
SHORT_DOC_THRESHOLD = 20   # ≤20页为短文档
MEDIUM_DOC_THRESHOLD = 50  # 21-50页为中等文档，>50为长文档

# 页面过滤模式
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
    depth = 0
    in_string = False
    escape = False
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


class AdaptivePageSelector:
    """自适应页面选择器"""
    
    def __init__(self, max_pages: int = 50, min_content_chars: int = 100):
        self.max_pages = max_pages
        self.min_content_chars = min_content_chars
    
    def _get_page_priority(self, text: str, page_num: int, total_pages: int) -> Tuple[str, int]:
        text_lower = text.lower().strip()
        first_lines = '\n'.join(text_lower.split('\n')[:5])
        
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                return ('skip', -1)
        
        clean_text = re.sub(r'\s+', '', text)
        if len(clean_text) < self.min_content_chars:
            return ('empty', -1)
        
        page_number_matches = re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)
        if len(page_number_matches) > 10:
            return ('toc', -1)
        
        priority = 0
        for pattern, score in PRIORITY_PATTERNS:
            if re.search(pattern, first_lines, re.IGNORECASE):
                priority = max(priority, score)
        
        # 前3页和最后3页加分
        if page_num <= 3 or page_num > total_pages - 3:
            priority += 3
        
        return ('normal', priority)
    
    def select_pages(self, doc: fitz.Document, doc_size: str) -> List[Tuple[int, str]]:
        """根据文档大小选择页面"""
        total_pages = len(doc)
        
        # 短文档和中等文档：返回所有有效页面
        if doc_size in ['short', 'medium']:
            pages = []
            for i in range(total_pages):
                text = doc[i].get_text()
                page_type, _ = self._get_page_priority(text, i + 1, total_pages)
                if page_type not in ['skip', 'empty', 'toc']:
                    pages.append((i, text))
            return pages
        
        # 长文档：智能选择
        page_info = []
        for i in range(total_pages):
            text = doc[i].get_text()
            page_type, priority = self._get_page_priority(text, i + 1, total_pages)
            if page_type not in ['skip', 'empty', 'toc']:
                page_info.append({
                    'index': i, 'text': text,
                    'priority': priority, 'char_count': len(text.strip())
                })
        
        if len(page_info) <= self.max_pages:
            return [(p['index'], p['text']) for p in page_info]
        
        selected = set()
        # 前5页
        for p in page_info[:5]:
            selected.add(p['index'])
        # 最后3页
        for p in page_info[-3:]:
            selected.add(p['index'])
        # 高优先级页面
        priority_pages = sorted([p for p in page_info if p['priority'] >= 5], key=lambda x: -x['priority'])
        for p in priority_pages[:15]:
            selected.add(p['index'])
        # 填充
        remaining = self.max_pages - len(selected)
        if remaining > 0:
            other = [p for p in page_info if p['index'] not in selected]
            other.sort(key=lambda x: -x['char_count'])
            for p in other[:remaining]:
                selected.add(p['index'])
        
        result = [(p['index'], p['text']) for p in page_info if p['index'] in selected]
        result.sort(key=lambda x: x[0])
        return result


class MedicalPDFExtractor:
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self.page_selector = AdaptivePageSelector()
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
    
    def _get_doc_size(self, total_pages: int) -> str:
        if total_pages <= SHORT_DOC_THRESHOLD:
            return 'short'
        elif total_pages <= MEDIUM_DOC_THRESHOLD:
            return 'medium'
        return 'long'
    
    def _extract_text(self, pdf_path: str) -> Tuple[str, Dict]:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc_size = self._get_doc_size(total_pages)
        
        selected_pages = self.page_selector.select_pages(doc, doc_size)
        
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
    
    def _build_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict], doc_size: str) -> str:
        # 根据文档大小选择不同的prompt策略
        if doc_size == 'short':
            # 短文档：详细提取
            base = f"""医学文献信息提取专家。从{doc_type}文档提取结构化信息。

要求：
1. sources必须标注具体页码如["p1"]或["p3-p5"]
2. 只提取原文存在的信息，绝对不编造
3. 提取具体完整内容，不是章节标题
4. 推荐/建议要完整引用原文表述
5. 尽可能提取所有关键信息

"""
        elif doc_size == 'medium':
            # 中等文档：标准提取
            base = f"""医学文献信息提取专家。从{doc_type}文档提取结构化信息。

要求：
1. sources标注页码如["p1"]或["p3-p5"]
2. 只提取原文存在的信息，不编造
3. 提取具体内容，不是章节标题
4. 推荐/建议要完整引用原文

"""
        else:
            # 长文档：摘要式提取
            base = f"""医学文献信息提取专家。从{doc_type}长文档中提取核心结构化信息。

注意：这是一份长文档，已智能选取关键页面，请：
1. sources标注页码如["p1"]或["p3-p5"]
2. 只提取原文存在的信息，不编造
3. 聚焦最重要的核心内容和结论
4. 推荐意见提取最关键的3-5条

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
        
        # 根据文档大小调整文本限制
        if doc_size == 'short':
            max_len = 12000
        elif doc_size == 'medium':
            max_len = 14000
        else:
            max_len = 14000
        
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
            doc_size = stats['doc_size']
            fewshot = self._load_fewshot(doc_type)
            prompt = self._build_prompt(doc_type, text, fewshot, doc_size)
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
        stats = result['stats']
        print(f"文档大小: {stats['doc_size']} ({stats['total_pages']}页→{stats['selected_pages']}页)")
    if result.get("success"):
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:2000])
