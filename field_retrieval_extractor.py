"""
方案5: 字段级检索增强提取器 (Field-centric Retrieval Extraction)
核心思想：
1. 全文提取 + 分chunk（带页码标记）
2. 按字段类型检索Top-K相关chunk
3. 每类字段独立调用LLM提取
4. Python确定性合并 + 去重
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import Counter
import math

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

# ============ 数据结构 ============
@dataclass
class TextChunk:
    """文本块"""
    index: int
    text: str
    pages: List[int]  # 涉及的页码
    char_count: int

# ============ 工具函数 ============
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


# ============ 分块器 ============
class PDFChunker:
    """PDF分块器：将PDF分成带页码标记的chunk"""
    
    def __init__(self, chunk_size: int = 6000, overlap: int = 500):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str]]:
        """提取所有页面文本，返回 [(页码, 文本), ...]"""
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:  # 跳过空白页
                pages.append((i + 1, text))
        doc.close()
        return pages
    
    def chunk_document(self, pdf_path: str) -> List[TextChunk]:
        """将PDF分成chunk"""
        pages = self.extract_pages(pdf_path)
        if not pages:
            return []
        
        # 合并所有页面文本，保留页码标记
        full_text = ""
        page_positions = []  # [(start_pos, end_pos, page_num), ...]
        
        for page_num, text in pages:
            start = len(full_text)
            full_text += f"\n[p{page_num}]\n{text}\n"
            end = len(full_text)
            page_positions.append((start, end, page_num))
        
        # 分chunk
        chunks = []
        pos = 0
        chunk_idx = 0
        
        while pos < len(full_text):
            end = min(pos + self.chunk_size, len(full_text))
            
            # 尝试在段落边界切分
            if end < len(full_text):
                for sep in ['\n\n', '\n', '. ', '。']:
                    last_sep = full_text.rfind(sep, pos + self.chunk_size // 2, end)
                    if last_sep > pos:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = full_text[pos:end]
            
            # 确定该chunk涉及哪些页码
            chunk_pages = []
            for start_p, end_p, page_num in page_positions:
                if start_p < end and end_p > pos:
                    chunk_pages.append(page_num)
            
            chunks.append(TextChunk(
                index=chunk_idx,
                text=chunk_text,
                pages=chunk_pages,
                char_count=len(chunk_text)
            ))
            
            chunk_idx += 1
            pos = end - self.overlap if end < len(full_text) else end
        
        return chunks


# ============ 简单检索器 (BM25-like) ============
class SimpleRetriever:
    """基于关键词的简单检索器"""
    
    # 各字段的检索关键词
    FIELD_KEYWORDS = {
        "metadata": [
            "title", "author", "abstract", "introduction", "doi", "journal",
            "标题", "作者", "摘要", "引言", "期刊", "年份", "published"
        ],
        "recommendations": [
            "recommend", "should", "must", "suggest", "advise", "guideline",
            "recommendation", "strong", "weak", "conditional", "grade",
            "推荐", "建议", "应该", "必须", "指南", "强推荐", "弱推荐",
            "level of evidence", "quality of evidence", "GRADE"
        ],
        "key_findings": [
            "result", "found", "showed", "demonstrated", "evidence", "significant",
            "outcome", "effect", "efficacy", "compared", "versus", "trial",
            "结果", "发现", "显示", "证据", "显著", "效果", "对比",
            "p-value", "CI", "confidence interval", "hazard ratio", "odds ratio"
        ],
        "conclusions": [
            "conclusion", "summary", "limitation", "implication", "future",
            "in summary", "in conclusion", "we conclude", "this study",
            "结论", "总结", "局限", "启示", "未来", "综上所述"
        ]
    }
    
    def __init__(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self._build_index()
    
    def _build_index(self):
        """构建词频索引"""
        self.doc_freq = Counter()  # 文档频率
        self.term_freq = []  # 每个chunk的词频
        
        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            tf = Counter()
            # 统计所有关键词出现次数
            for field, keywords in self.FIELD_KEYWORDS.items():
                for kw in keywords:
                    count = text_lower.count(kw.lower())
                    if count > 0:
                        tf[kw.lower()] = count
                        self.doc_freq[kw.lower()] += 1
            self.term_freq.append(tf)
    
    def retrieve(self, field: str, top_k: int = 3) -> List[TextChunk]:
        """检索与指定字段最相关的chunk"""
        keywords = self.FIELD_KEYWORDS.get(field, [])
        if not keywords:
            return self.chunks[:top_k]
        
        scores = []
        n_docs = len(self.chunks)
        
        for i, chunk in enumerate(self.chunks):
            score = 0.0
            tf = self.term_freq[i]
            
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in tf:
                    # BM25-like scoring
                    term_f = tf[kw_lower]
                    doc_f = self.doc_freq.get(kw_lower, 1)
                    idf = math.log((n_docs - doc_f + 0.5) / (doc_f + 0.5) + 1)
                    score += term_f * idf
            
            scores.append((score, i, chunk))
        
        # 按分数排序，取top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 返回分数>0的chunk，最多top_k个
        result = [item[2] for item in scores[:top_k] if item[0] > 0]
        
        # 如果没有匹配，返回首尾chunk
        if not result:
            result = [self.chunks[0]]
            if len(self.chunks) > 1:
                result.append(self.chunks[-1])
        
        return result


# ============ 字段级提取器 ============
class FieldExtractor:
    """字段级提取器：针对每类字段独立提取"""
    
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
        self._fewshot_cache = {}
    
    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        resp = requests.post(
            self.api_url,
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=180
        )
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def extract_metadata(self, chunks: List[TextChunk]) -> Dict:
        """提取元数据"""
        text = "\n---\n".join([c.text for c in chunks])[:8000]
        
        prompt = f"""从以下医学文献片段提取基本信息。

文本：
{text}

返回JSON格式：
{{
  "title": "文档标题",
  "authors": "作者列表",
  "organization": "发布机构",
  "publish_date": "发布年份",
  "document_type": "文档类型(指南/综述/研究等)",
  "doi": "DOI号(如有)",
  "sources": ["p1"]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=800)
        json_str = extract_first_json(result)
        if json_str:
            try:
                return json.loads(json_str)
            except:
                pass
        return {"title": "Unknown", "sources": []}
    
    def extract_recommendations(self, chunks: List[TextChunk], doc_type: str) -> List[Dict]:
        """提取推荐意见（主要用于GUIDELINE类型）"""
        if doc_type != "GUIDELINE":
            return []
        
        text = "\n---\n".join([c.text for c in chunks])[:10000]
        
        prompt = f"""从以下临床指南片段提取推荐意见。

文本：
{text}

要求：
1. 提取具体的临床推荐，不是章节标题
2. 保留推荐强度(强推荐/弱推荐/条件推荐等)
3. 标注来源页码

返回JSON格式：
{{
  "recommendations": [
    {{
      "id": "R1",
      "text": "完整推荐内容原文",
      "strength": "推荐强度",
      "evidence_level": "证据等级(如有)",
      "sources": ["px"]
    }}
  ]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=3000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                return data.get("recommendations", [])
            except:
                pass
        return []
    
    def extract_findings(self, chunks: List[TextChunk]) -> List[Dict]:
        """提取关键发现"""
        text = "\n---\n".join([c.text for c in chunks])[:10000]
        
        prompt = f"""从以下医学文献片段提取关键研究发现。

文本：
{text}

要求：
1. 提取具体的研究结果和发现
2. 包含数据时保留具体数值(p值、CI、效应量等)
3. 标注来源页码

返回JSON格式：
{{
  "key_findings": [
    {{
      "id": "F1",
      "finding": "具体发现内容",
      "data": "相关数据(如有)",
      "sources": ["px"]
    }}
  ]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=3000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                return data.get("key_findings", [])
            except:
                pass
        return []
    
    def extract_conclusions(self, chunks: List[TextChunk]) -> List[Dict]:
        """提取结论"""
        text = "\n---\n".join([c.text for c in chunks])[:8000]
        
        prompt = f"""从以下医学文献片段提取结论和总结。

文本：
{text}

要求：
1. 提取主要结论
2. 包含局限性说明(如有)
3. 标注来源页码

返回JSON格式：
{{
  "conclusions": [
    {{
      "conclusion": "结论内容",
      "sources": ["px"]
    }}
  ],
  "limitations": "研究局限性(如有)"
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=2000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                return data.get("conclusions", [])
            except:
                pass
        return []


# ============ 主提取器 ============
class FieldRetrievalExtractor:
    """方案5: 字段级检索增强提取器"""
    
    def __init__(self, api_url: str = LOCAL_API, chunk_size: int = 6000, top_k: int = 4):
        self.api_url = api_url
        self.chunker = PDFChunker(chunk_size=chunk_size)
        self.field_extractor = FieldExtractor(api_url)
        self.top_k = top_k
    
    def classify(self, text: str) -> str:
        """分类文档类型"""
        prompt = """判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER
- GUIDELINE: 临床指南/诊疗规范/技术评估
- REVIEW: 综述/系统评价/Meta分析
- OTHER: 其他

文档开头：
""" + text[:3000] + """

只返回类型名："""
        
        resp = requests.post(
            self.api_url,
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "temperature": 0.1,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=60
        )
        result = resp.json()["choices"][0]["message"]["content"]
        
        for t in ["GUIDELINE", "REVIEW", "OTHER"]:
            if t in result.upper():
                return t
        return "OTHER"
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """主提取流程"""
        start_time = time.time()
        
        try:
            # Step 1: 分chunk
            chunks = self.chunker.chunk_document(pdf_path)
            if not chunks:
                return {"success": False, "error": "PDF内容为空", "time": time.time() - start_time}
            
            total_pages = max(max(c.pages) for c in chunks if c.pages)
            print(f"[方案5] 文档共{total_pages}页，分成{len(chunks)}个chunk")
            
            # Step 2: 分类
            first_chunk_text = chunks[0].text if chunks else ""
            doc_type = self.classify(first_chunk_text)
            print(f"[方案5] 文档类型: {doc_type}")
            
            # Step 3: 构建检索器
            retriever = SimpleRetriever(chunks)
            
            # Step 4: 按字段检索并提取
            results = {"doc_type": doc_type}
            
            # 4.1 元数据 - 主要从前面的chunk
            metadata_chunks = retriever.retrieve("metadata", top_k=2)
            print(f"[方案5] 元数据检索: {[c.pages for c in metadata_chunks]}")
            results["doc_metadata"] = self.field_extractor.extract_metadata(metadata_chunks)
            
            # 4.2 推荐意见 (仅GUIDELINE)
            if doc_type == "GUIDELINE":
                rec_chunks = retriever.retrieve("recommendations", top_k=self.top_k)
                print(f"[方案5] 推荐意见检索: {[c.pages for c in rec_chunks]}")
                results["recommendations"] = self.field_extractor.extract_recommendations(rec_chunks, doc_type)
            
            # 4.3 关键发现
            findings_chunks = retriever.retrieve("key_findings", top_k=self.top_k)
            print(f"[方案5] 关键发现检索: {[c.pages for c in findings_chunks]}")
            results["key_findings"] = self.field_extractor.extract_findings(findings_chunks)
            
            # 4.4 结论
            conclusions_chunks = retriever.retrieve("conclusions", top_k=self.top_k)
            print(f"[方案5] 结论检索: {[c.pages for c in conclusions_chunks]}")
            results["conclusions"] = self.field_extractor.extract_conclusions(conclusions_chunks)
            
            # Step 5: 统计覆盖的页码
            all_covered_pages = set()
            for c in metadata_chunks + findings_chunks + conclusions_chunks:
                all_covered_pages.update(c.pages)
            if doc_type == "GUIDELINE":
                for c in rec_chunks:
                    all_covered_pages.update(c.pages)
            
            return {
                "success": True,
                "doc_type": doc_type,
                "result": results,
                "time": time.time() - start_time,
                "stats": {
                    "total_pages": total_pages,
                    "total_chunks": len(chunks),
                    "covered_pages": sorted(all_covered_pages),
                    "coverage_ratio": len(all_covered_pages) / total_pages if total_pages > 0 else 0
                }
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False, 
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "time": time.time() - start_time
            }


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    """便捷函数"""
    return FieldRetrievalExtractor().extract(pdf_path)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    print(f"\n=== 结果 ===")
    print(f"成功: {result.get('success')}")
    print(f"类型: {result.get('doc_type')}")
    print(f"耗时: {result.get('time', 0):.1f}s")
    if result.get("stats"):
        stats = result["stats"]
        print(f"覆盖页码: {stats.get('covered_pages', [])}")
        print(f"覆盖率: {stats.get('coverage_ratio', 0)*100:.1f}%")
    if result.get("success"):
        print(f"\n提取结果预览:")
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:2000])
