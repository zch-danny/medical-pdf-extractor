"""
生产级医学PDF结构化提取器 v7.10
优化：精简Few-shot + 结构化中文提示词 + 明确指令
"""
import json, re, time, requests, fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

LOCAL_API = "http://localhost:8000/v1/chat/completions"

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

def clean_json_string(s):
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text):
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    text = clean_json_string(text)
    start = text.find('{')
    if start == -1: return None
    depth, in_string, escape = 0, False, False
    for i, c in enumerate(text[start:], start):
        if escape: escape = False; continue
        if c == '\\': escape = True; continue
        if c == '"' and not escape: in_string = not in_string; continue
        if in_string: continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None

class HybridPageSelector:
    def __init__(self, max_pages=50, min_chars=100):
        self.max_pages = max_pages
        self.min_chars = min_chars
    
    def _is_empty(self, text): return len(re.sub(r'\s+', '', text)) < self.min_chars
    
    def _should_skip(self, text):
        first = '\n'.join(text.lower().strip().split('\n')[:5])
        for p in SKIP_PATTERNS:
            if re.search(p, first, re.IGNORECASE): return True
        if len(re.findall(r'\.\s*\d{1,3}\s*$', text, re.MULTILINE)) > 10: return True
        return False
    
    def _priority(self, text, num, total):
        first = '\n'.join(text.lower().split('\n')[:5])
        p = 0
        for pat, score in PRIORITY_PATTERNS:
            if re.search(pat, first, re.IGNORECASE): p = max(p, score)
        if num <= 3 or num > total - 3: p += 3
        return p
    
    def select(self, doc):
        total = len(doc)
        doc_size = 'short' if total <= SHORT_DOC_THRESHOLD else ('medium' if total <= MEDIUM_DOC_THRESHOLD else 'long')
        
        pages = []
        for i in range(total):
            text = doc[i].get_text()
            if self._is_empty(text): continue
            if doc_size != 'short' and self._should_skip(text): continue
            pages.append({'idx': i, 'text': text, 'priority': self._priority(text, i+1, total)})
        
        if doc_size == 'long' and len(pages) > self.max_pages:
            sel = set(p['idx'] for p in pages[:5])
            sel.update(p['idx'] for p in pages[-3:])
            for p in sorted(pages, key=lambda x: -x['priority']):
                if len(sel) >= self.max_pages: break
                sel.add(p['idx'])
            pages = [p for p in pages if p['idx'] in sel]
            pages.sort(key=lambda x: x['idx'])
        
        return [(p['idx'], p['text']) for p in pages], doc_size

# 精简的中文Few-shot示例
FEWSHOT = {
    "GUIDELINE": '''{"doc_metadata":{"title":"心力衰竭诊疗指南","organization":"ESC","year":"2025"},"scope":{"objective":"心衰诊断和治疗","population":"成人心衰患者"},"recommendations":[{"id":"1.1","text":"推荐所有HFrEF患者使用ACE抑制剂以降低死亡率和住院率","strength":"I类推荐,A级证据","sources":["p12"]}],"key_evidence":[{"evidence":"SOLVD试验显示ACE抑制剂降低死亡率16%","sources":["p15"]}]}''',
    "REVIEW": '''{"doc_metadata":{"title":"SGLT2抑制剂荟萃分析","journal":"Lancet","year":"2025"},"scope":{"objective":"评估SGLT2抑制剂心血管获益"},"key_findings":[{"id":"F1","finding":"SGLT2抑制剂降低心血管死亡23%(HR 0.77, 95%CI 0.71-0.84)","sources":["p8"]}],"conclusions":{"main":"SGLT2抑制剂显著降低心血管事件","limitations":"纳入研究异质性中等"}}''',
    "OTHER": '''{"doc_metadata":{"title":"急性肾损伤新型生物标志物","year":"2025"},"scope":{"objective":"评估NGAL预测AKI价值"},"key_findings":[{"id":"F1","finding":"NGAL>150ng/mL预测AKI敏感性89%,特异性78%","sources":["p5"]}],"conclusions":{"main":"NGAL是有前景的AKI早期标志物"}}'''
}

# 结构化提示词
EXTRACT_PROMPTS = {
    "GUIDELINE": '''你是医学文献信息提取专家。请从以下临床指南中提取结构化信息。

【提取要求】
1. doc_metadata: 提取标题、作者/组织、发布年份
2. scope: 提取指南目的和目标人群
3. recommendations: 提取所有推荐建议，必须包含：
   - 完整的推荐原文（不要概括）
   - 推荐强度(如Class I/II/III, Level A/B/C)
   - 来源页码(格式["p1"]或["p3-p5"])
4. key_evidence: 提取支持推荐的关键证据

【输出示例】
{example}

【文档内容】
{text}

【输出JSON】''',

    "REVIEW": '''你是医学文献信息提取专家。请从以下综述/荟萃分析中提取结构化信息。

【提取要求】
1. doc_metadata: 提取标题、作者、期刊、年份
2. scope: 提取研究目的
3. key_findings: 提取所有主要发现，必须包含：
   - 具体的研究发现（含数据如HR、OR、CI等）
   - 来源页码
4. conclusions: 提取主要结论和局限性

【输出示例】
{example}

【文档内容】
{text}

【输出JSON】''',

    "OTHER": '''你是医学文献信息提取专家。请从以下医学文档中提取结构化信息。

【提取要求】
1. doc_metadata: 提取标题、作者、年份
2. scope: 提取研究目的
3. key_findings: 提取主要发现（含具体数据）
4. conclusions: 提取主要结论

【输出示例】
{example}

【文档内容】
{text}

【输出JSON】'''
}

class MedicalPDFExtractor:
    def __init__(self, api_url=LOCAL_API):
        self.api_url = api_url
        self.selector = HybridPageSelector()
    
    def _llm(self, prompt, max_tokens=6000):
        r = requests.post(self.api_url, json={
            "model": "qwen3-8b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}
        }, timeout=300)
        d = r.json()
        if "choices" not in d: raise Exception(f"API error: {d}")
        return d["choices"][0]["message"]["content"]
    
    def _extract_text(self, path):
        doc = fitz.open(path)
        pages, doc_size = self.selector.select(doc)
        text = "\n".join([f"=== 第{i+1}页 ===\n{t}" for i, t in pages])
        stats = {'total_pages': len(doc), 'selected_pages': len(pages), 'doc_size': doc_size}
        doc.close()
        return text, stats
    
    def classify(self, text):
        prompt = f"""判断以下医学文档的类型，返回GUIDELINE、REVIEW或OTHER中的一个：
- GUIDELINE: 临床指南、诊疗规范、技术评估报告
- REVIEW: 系统综述、荟萃分析、文献综述
- OTHER: 原始研究、病例报告、其他文档

文档开头：
{text[:2500]}

只返回类型名称："""
        r = self._llm(prompt, 50)
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in r.upper(): return t
        return "OTHER"
    
    def extract(self, path):
        start = time.time()
        try:
            text, stats = self._extract_text(path)
            if len(text.strip()) < 100:
                return {"success": False, "error": "内容过少", "time": time.time()-start}
            
            doc_type = self.classify(text)
            
            if len(text) > 12000:
                text = text[:6000] + "\n\n...[中间内容省略]...\n\n" + text[-6000:]
            
            prompt = EXTRACT_PROMPTS[doc_type].format(example=FEWSHOT[doc_type], text=text)
            result = self._llm(prompt)
            json_str = extract_first_json(result)
            
            if json_str:
                data = json.loads(json_str)
                return {"success": True, "doc_type": doc_type, "result": data, "time": time.time()-start, "stats": stats}
            return {"success": False, "error": "未找到JSON", "time": time.time()-start}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time()-start}

def extract_pdf(path):
    return MedicalPDFExtractor().extract(path)

if __name__ == "__main__":
    import sys
    r = extract_pdf(sys.argv[1] if len(sys.argv) > 1 else "example.pdf")
    print(f"类型: {r.get('doc_type')}, 成功: {r.get('success')}, 耗时: {r.get('time',0):.1f}s")
