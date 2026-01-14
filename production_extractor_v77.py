"""
生产级医学PDF结构化提取器 v7.7
融合版：v7.6混合页面选择 + 方案5字段级检索
- 短文档(≤15页): 全部保留
- 中等文档(16-50页): 跳过目录/参考文献
- 长文档(>50页): 智能选择50页 + 字段级检索
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# 字段检索关键词
FIELD_KEYWORDS = {
    'metadata': ['title', 'author', 'journal', 'doi', 'abstract', '标题', '作者', '摘要', 'published', 'copyright'],
    'recommendations': ['recommend', 'should', 'must', 'class i', 'class ii', 'level a', 'level b', 
                       '推荐', '建议', '应该', '必须', 'strong', 'weak', 'conditional', 'grade'],
    'findings': ['found', 'showed', 'demonstrated', 'result', 'significant', 'p<', 'ci ', 'or ', 'rr ',
                '发现', '显示', '结果', '表明', 'outcome', 'hazard', 'ratio'],
    'conclusions': ['conclusion', 'summary', 'in summary', 'overall', '结论', '总结', '综上'],
}

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

@dataclass
class PageContent:
    index: int
    text: str
    priority: int = 0
    field_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.field_scores is None:
            self.field_scores = {}

class HybridExtractor:
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
    
    def _is_empty_page(self, text: str, min_chars: int = 100) -> bool:
        return len(re.sub(r'\s+', '', text)) < min_chars
    
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
    
    def _score_page_for_field(self, text: str, field: str) -> float:
        """计算页面对特定字段的相关性分数"""
        text_lower = text.lower()
        keywords = FIELD_KEYWORDS.get(field, [])
        score = 0
        for kw in keywords:
            count = len(re.findall(re.escape(kw), text_lower))
            score += min(count, 5)  # 每个关键词最多计5次
        return score
    
    def _select_pages(self, doc: fitz.Document) -> Tuple[List[PageContent], str, Dict]:
        """混合页面选择策略"""
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
            page = PageContent(index=i, text=text, priority=priority)
            
            # 为长文档计算字段相关性分数
            if doc_size == 'long':
                for field in FIELD_KEYWORDS:
                    page.field_scores[field] = self._score_page_for_field(text, field)
            
            pages.append(page)
        
        # 长文档：智能选择最多50页
        if doc_size == 'long' and len(pages) > 50:
            selected = set()
            # 保留前5页和后3页
            for p in pages[:5]:
                selected.add(p.index)
            for p in pages[-3:]:
                selected.add(p.index)
            # 按优先级选择高分页
            for p in sorted(pages, key=lambda x: -x.priority)[:15]:
                selected.add(p.index)
            # 按内容长度选择
            remaining = 50 - len(selected)
            if remaining > 0:
                for p in sorted([x for x in pages if x.index not in selected], 
                               key=lambda x: -len(x.text))[:remaining]:
                    selected.add(p.index)
            pages = [p for p in pages if p.index in selected]
            pages.sort(key=lambda x: x.index)
        
        stats = {
            'total_pages': total_pages,
            'selected_pages': len(pages),
            'doc_size': doc_size
        }
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
        prompt = "判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER\n- GUIDELINE: 临床指南/诊疗规范/技术评估\n- REVIEW: 综述/系统评价/Meta分析\n- OTHER: 其他\n\n文档开头：\n" + text[:2500] + "\n\n只返回类型名："
        result = self._call_llm(prompt, max_tokens=50, timeout=60)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def _get_field_relevant_text(self, pages: List[PageContent], field: str, doc_size: str, max_chars: int = 8000) -> str:
        """获取与特定字段最相关的文本"""
        if doc_size == 'short':
            # 短文档直接返回全部
            text = "\n".join([f"=== 第{p.index+1}页 ===\n{p.text}" for p in pages])
            return text[:max_chars] if len(text) > max_chars else text
        
        if doc_size == 'long' and field in FIELD_KEYWORDS:
            # 长文档按字段相关性排序
            sorted_pages = sorted(pages, key=lambda x: -x.field_scores.get(field, 0))
            top_pages = sorted_pages[:8]  # 取最相关的8页
            top_pages.sort(key=lambda x: x.index)  # 按页码顺序
        else:
            top_pages = pages[:15]
        
        text = "\n".join([f"=== 第{p.index+1}页 ===\n{p.text}" for p in top_pages])
        return text[:max_chars] if len(text) > max_chars else text
    
    def _build_unified_prompt(self, doc_type: str, text: str, fewshot: Optional[Dict]) -> str:
        """构建统一提取prompt"""
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
        
        if len(text) > 12000:
            text = text[:6000] + "\n\n...[中间部分省略]...\n\n" + text[-6000:]
        
        return f"{base}{fewshot_section}格式：{fmt}\n\n文档：\n{text}\n\n返回JSON："
    
    def _extract_field_parallel(self, pages: List[PageContent], doc_type: str, doc_size: str) -> Dict:
        """并行提取各字段（仅用于长文档）"""
        if doc_size != 'long':
            return None
        
        results = {}
        fields_to_extract = ['metadata', 'recommendations' if doc_type == 'GUIDELINE' else 'findings', 'conclusions']
        
        def extract_single_field(field: str) -> Tuple[str, Any]:
            relevant_text = self._get_field_relevant_text(pages, field, doc_size, max_chars=6000)
            
            if field == 'metadata':
                prompt = f"从医学文献提取元数据，返回JSON：\n{relevant_text[:3000]}\n\n格式：{{\"title\":\"...\",\"authors\":[...],\"journal\":\"...\",\"year\":\"...\",\"doi\":\"...\"}}"
            elif field == 'recommendations':
                prompt = f"从指南文献提取推荐建议，返回JSON数组：\n{relevant_text}\n\n格式：[{{\"id\":\"1.1\",\"text\":\"推荐内容\",\"strength\":\"强度\",\"sources\":[\"px\"]}}]"
            elif field == 'findings':
                prompt = f"从文献提取主要发现，返回JSON数组：\n{relevant_text}\n\n格式：[{{\"id\":\"F1\",\"finding\":\"发现内容\",\"sources\":[\"px\"]}}]"
            elif field == 'conclusions':
                prompt = f"从文献提取结论，返回JSON：\n{relevant_text}\n\n格式：{{\"main_conclusion\":\"...\",\"clinical_implications\":\"...\"}}"
            else:
                return field, None
            
            try:
                result = self._call_llm(prompt, max_tokens=3000, timeout=120)
                json_str = extract_first_json(result)
                if json_str:
                    return field, json.loads(json_str)
            except Exception as e:
                print(f"[警告] {field}提取失败: {e}")
            return field, None
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(extract_single_field, f): f for f in fields_to_extract}
            for future in as_completed(futures):
                field, data = future.result()
                if data:
                    results[field] = data
        
        return results if results else None
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages, doc_size, stats = self._select_pages(doc)
            doc.close()
            
            if not pages:
                return {"success": False, "error": "PDF内容过少", "time": time.time() - start_time, "stats": stats}
            
            # 分类
            classify_text = "\n".join([p.text for p in pages[:3]])[:2500]
            doc_type = self.classify(classify_text)
            
            # 对于长文档，尝试并行字段提取
            parallel_results = None
            if doc_size == 'long':
                parallel_results = self._extract_field_parallel(pages, doc_type, doc_size)
            
            # 统一提取（作为主要结果或补充）
            full_text = self._get_field_relevant_text(pages, 'all', doc_size)
            fewshot = self._load_fewshot(doc_type)
            prompt = self._build_unified_prompt(doc_type, full_text, fewshot)
            result = self._call_llm(prompt)
            
            json_str = extract_first_json(result)
            if json_str:
                data = json.loads(json_str)
                
                # 合并并行提取的结果（长文档）
                if parallel_results:
                    if 'metadata' in parallel_results and parallel_results['metadata']:
                        data['doc_metadata'] = {**data.get('doc_metadata', {}), **parallel_results['metadata']}
                    if 'recommendations' in parallel_results and doc_type == 'GUIDELINE':
                        existing = data.get('recommendations', [])
                        new_recs = parallel_results['recommendations']
                        if isinstance(new_recs, list) and len(new_recs) > len(existing):
                            data['recommendations'] = new_recs
                    if 'findings' in parallel_results and doc_type != 'GUIDELINE':
                        existing = data.get('key_findings', [])
                        new_findings = parallel_results['findings']
                        if isinstance(new_findings, list) and len(new_findings) > len(existing):
                            data['key_findings'] = new_findings
                    if 'conclusions' in parallel_results:
                        data['conclusions'] = {**data.get('conclusions', {}), **parallel_results['conclusions']}
                
                return {"success": True, "doc_type": doc_type, "result": data, "time": time.time() - start_time, "stats": stats}
            else:
                return {"success": False, "error": "未找到有效JSON", "time": time.time() - start_time, "stats": stats}
        
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON解析失败: {e}", "time": time.time() - start_time}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}

def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    return HybridExtractor().extract(pdf_path)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"类型: {result.get('doc_type')}, 成功: {result.get('success')}, 耗时: {result.get('time', 0):.1f}s")
    if 'stats' in result:
        s = result['stats']
        print(f"文档: {s['doc_size']} ({s['total_pages']}页→{s['selected_pages']}页)")
