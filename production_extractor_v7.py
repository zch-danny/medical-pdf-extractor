"""
生产级医学PDF结构化提取器 v7.2
改进：处理超长文档 + 修复JSON控制字符问题
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

def clean_json_string(s: str) -> str:
    """清理JSON字符串中的控制字符"""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text: str) -> Optional[str]:
    """提取第一个完整的JSON对象"""
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


class MedicalPDFExtractor:
    def __init__(self, api_url: str = LOCAL_API, max_pages: int = 15):
        self.api_url = api_url
        self.max_pages = max_pages
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
    
    def _extract_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            if i >= self.max_pages:
                break
            pages.append(f"=== 第{i+1}页 ===\n{page.get_text()}")
        doc.close()
        return "\n".join(pages)
    
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
        
        max_len = 10000
        if len(text) > max_len:
            text = text[:max_len] + "\n...[截断]"
        
        return f"{base}{fewshot_section}格式：{fmt}\n\n文档：\n{text}\n\n返回JSON："

    def extract(self, pdf_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            text = self._extract_text(pdf_path)
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
                    "time": time.time() - start_time
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
    if result.get("success"):
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:1500])