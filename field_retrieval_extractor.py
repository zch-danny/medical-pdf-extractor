"""
方案5优化版: 字段级检索增强提取器 v2
优化：
1. 增加top_k提高覆盖率
2. 增强医学领域检索关键词
3. 添加scope字段提取
4. 结果去重和质量优化
5. 智能首尾页保留
"""
import json
import re
import time
import requests
import fitz
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
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
    pages: List[int]
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

def deduplicate_findings(findings: List[Dict], threshold: float = 0.7) -> List[Dict]:
    """去重：基于文本相似度"""
    if not findings:
        return []
    
    result = []
    seen_texts = []
    
    for item in findings:
        text = item.get("finding", item.get("conclusion", item.get("text", "")))
        if not text:
            continue
        
        # 简单去重：检查是否有高度相似的已存在文本
        is_dup = False
        text_lower = text.lower()
        for seen in seen_texts:
            # 计算简单的重叠度
            words1 = set(text_lower.split())
            words2 = set(seen.split())
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                if overlap > threshold:
                    is_dup = True
                    break
        
        if not is_dup:
            result.append(item)
            seen_texts.append(text_lower)
    
    return result


# ============ 智能分块器 ============
class SmartPDFChunker:
    """智能PDF分块器：跳过无用页，保留关键页"""
    
    # 应该跳过的页面模式
    SKIP_PATTERNS = [
        r'^table\s+of\s+contents?$',
        r'^contents?$',
        r'^目录$',
        r'^references?$',
        r'^bibliography$',
        r'^参考文献$',
        r'^appendix',
        r'^附录',
        r'^index$',
        r'^acknowledgement',
        r'^致谢',
    ]
    
    def __init__(self, chunk_size: int = 5000, overlap: int = 400):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._skip_re = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]
    
    def _should_skip_page(self, text: str) -> bool:
        """判断是否应跳过该页"""
        # 空白页或内容过少
        if len(text.strip()) < 100:
            return True
        
        # 检查首行是否匹配跳过模式
        first_line = text.strip().split('\n')[0].strip()
        for pattern in self._skip_re:
            if pattern.match(first_line):
                return True
        
        # 参考文献页特征：大量的年份和作者格式
        ref_pattern = r'\(\d{4}\)|\d{4}[;,]'
        if len(re.findall(ref_pattern, text)) > 20:
            return True
        
        return False
    
    def extract_pages(self, pdf_path: str) -> List[Tuple[int, str, bool]]:
        """提取所有页面，标记是否为关键页"""
        doc = fitz.open(pdf_path)
        pages = []
        
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            page_num = i + 1
            
            # 标记关键页：首页、尾页、摘要页等
            is_key = (
                page_num <= 3 or  # 前3页（标题、摘要）
                page_num >= len(doc) - 2 or  # 最后3页（结论）
                'abstract' in text.lower()[:500] or
                'summary' in text.lower()[:500] or
                '摘要' in text[:500]
            )
            
            if text and not self._should_skip_page(text):
                pages.append((page_num, text, is_key))
        
        doc.close()
        return pages
    
    def chunk_document(self, pdf_path: str) -> Tuple[List[TextChunk], Dict]:
        """将PDF分成chunk，返回chunks和统计信息"""
        pages = self.extract_pages(pdf_path)
        if not pages:
            return [], {}
        
        # 统计
        total_doc_pages = fitz.open(pdf_path).page_count
        kept_pages = len(pages)
        key_pages = [p[0] for p in pages if p[2]]
        
        # 合并文本
        full_text = ""
        page_positions = []
        
        for page_num, text, is_key in pages:
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
            
            if end < len(full_text):
                for sep in ['\n\n', '\n', '. ', '。']:
                    last_sep = full_text.rfind(sep, pos + self.chunk_size // 2, end)
                    if last_sep > pos:
                        end = last_sep + len(sep)
                        break
            
            chunk_text = full_text[pos:end]
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
        
        stats = {
            "total_doc_pages": total_doc_pages,
            "kept_pages": kept_pages,
            "skipped_pages": total_doc_pages - kept_pages,
            "key_pages": key_pages,
            "total_chunks": len(chunks)
        }
        
        return chunks, stats


# ============ 增强检索器 ============
class EnhancedRetriever:
    """增强版检索器：更多医学领域关键词"""
    
    FIELD_KEYWORDS = {
        "metadata": [
            # 英文
            "title", "author", "abstract", "introduction", "doi", "journal",
            "published", "copyright", "correspondence", "affiliation",
            "background", "objective", "purpose", "aim",
            # 中文
            "标题", "作者", "摘要", "引言", "期刊", "年份", "背景", "目的"
        ],
        "scope": [
            # 英文
            "scope", "objective", "aim", "purpose", "target", "population",
            "inclusion", "exclusion", "criteria", "setting", "context",
            "methods", "methodology", "study design", "approach",
            # 中文
            "范围", "目标", "目的", "纳入", "排除", "标准", "方法", "设计"
        ],
        "recommendations": [
            # 英文 - 推荐相关
            "recommend", "recommendation", "should", "must", "suggest",
            "advise", "guideline", "guidance", "statement",
            "strong recommendation", "weak recommendation", "conditional",
            "grade", "level of evidence", "quality of evidence",
            "class i", "class ii", "class iii", "level a", "level b", "level c",
            # 中文
            "推荐", "建议", "应该", "必须", "指南", "强推荐", "弱推荐",
            "证据等级", "推荐强度", "一级推荐", "二级推荐"
        ],
        "key_findings": [
            # 英文 - 结果相关
            "result", "finding", "found", "showed", "demonstrated",
            "evidence", "significant", "outcome", "effect", "efficacy",
            "compared", "versus", "trial", "study", "analysis",
            "p-value", "p value", "p <", "p =", "ci", "confidence interval",
            "hazard ratio", "odds ratio", "risk ratio", "relative risk",
            "absolute risk", "number needed", "nnt", "nntt",
            "mean difference", "standardized mean", "forest plot",
            "heterogeneity", "i2", "i²", "meta-analysis",
            # 数据模式
            "95%", "0.05", "0.01", "0.001",
            # 中文
            "结果", "发现", "显示", "证据", "显著", "效果", "对比",
            "风险比", "优势比", "置信区间", "异质性"
        ],
        "conclusions": [
            # 英文
            "conclusion", "conclusions", "summary", "in summary",
            "in conclusion", "we conclude", "this study", "our findings",
            "limitation", "limitations", "weakness", "strength",
            "implication", "implications", "future", "further research",
            "clinical significance", "practical implications",
            # 中文
            "结论", "总结", "综上所述", "局限", "局限性", "启示",
            "未来研究", "临床意义", "实践意义"
        ]
    }
    
    def __init__(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self._build_index()
    
    def _build_index(self):
        """构建词频索引"""
        self.doc_freq = Counter()
        self.term_freq = []
        
        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            tf = Counter()
            for field, keywords in self.FIELD_KEYWORDS.items():
                for kw in keywords:
                    count = text_lower.count(kw.lower())
                    if count > 0:
                        tf[kw.lower()] = count
                        self.doc_freq[kw.lower()] += 1
            self.term_freq.append(tf)
    
    def retrieve(self, field: str, top_k: int = 5, include_first_last: bool = True) -> List[TextChunk]:
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
                    term_f = tf[kw_lower]
                    doc_f = self.doc_freq.get(kw_lower, 1)
                    idf = math.log((n_docs - doc_f + 0.5) / (doc_f + 0.5) + 1)
                    score += term_f * idf
            
            # 对首尾chunk加权（更可能包含摘要和结论）
            if include_first_last:
                if i == 0:
                    score *= 1.3
                elif i == n_docs - 1:
                    score *= 1.2
            
            scores.append((score, i, chunk))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # 取top_k个分数>0的chunk
        result = []
        seen_indices = set()
        
        for score, idx, chunk in scores:
            if score > 0 and idx not in seen_indices and len(result) < top_k:
                result.append(chunk)
                seen_indices.add(idx)
        
        # 确保至少包含首尾chunk
        if include_first_last and len(self.chunks) > 1:
            if 0 not in seen_indices and len(result) < top_k + 1:
                result.insert(0, self.chunks[0])
            if (n_docs - 1) not in seen_indices and len(result) < top_k + 2:
                result.append(self.chunks[-1])
        
        return result if result else [self.chunks[0]]


# ============ 增强字段提取器 ============
class EnhancedFieldExtractor:
    """增强版字段提取器"""
    
    def __init__(self, api_url: str = LOCAL_API):
        self.api_url = api_url
    
    def _call_llm(self, prompt: str, max_tokens: int = 2500) -> str:
        resp = requests.post(
            self.api_url,
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.15,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=200
        )
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def extract_metadata(self, chunks: List[TextChunk]) -> Dict:
        """提取元数据"""
        text = "\n---\n".join([c.text for c in chunks])[:10000]
        
        prompt = f"""从以下医学文献片段提取基本信息。

文本：
{text}

返回JSON格式：
{{
  "title": "文档完整标题",
  "authors": "作者列表(用逗号分隔)",
  "organization": "发布机构/期刊名",
  "publish_date": "发布年份",
  "document_type": "文档类型(指南/综述/研究/手册等)",
  "doi": "DOI号(如有)",
  "language": "语言",
  "sources": ["p1"]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=1000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                return json.loads(json_str)
            except:
                pass
        return {"title": "Unknown", "sources": []}
    
    def extract_scope(self, chunks: List[TextChunk]) -> Dict:
        """提取范围和目标"""
        text = "\n---\n".join([c.text for c in chunks])[:8000]
        
        prompt = f"""从以下医学文献片段提取研究范围和目标。

文本：
{text}

返回JSON格式：
{{
  "objectives": "研究目标/目的",
  "target_population": "目标人群(如有)",
  "setting": "研究场景/环境(如有)",
  "methods_summary": "方法概述",
  "sources": ["px"]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=1200)
        json_str = extract_first_json(result)
        if json_str:
            try:
                return json.loads(json_str)
            except:
                pass
        return {"objectives": "", "sources": []}
    
    def extract_recommendations(self, chunks: List[TextChunk], doc_type: str) -> List[Dict]:
        """提取推荐意见"""
        if doc_type != "GUIDELINE":
            return []
        
        text = "\n---\n".join([c.text for c in chunks])[:12000]
        
        prompt = f"""从以下临床指南片段提取所有推荐意见。

文本：
{text}

要求：
1. 提取具体的临床推荐，不是章节标题
2. 保留推荐强度(强/弱/条件推荐)和证据等级
3. 完整引用推荐原文
4. 标注来源页码

返回JSON格式：
{{
  "recommendations": [
    {{
      "id": "R1",
      "text": "完整推荐内容原文",
      "strength": "推荐强度(强推荐/弱推荐/条件推荐等)",
      "evidence_level": "证据等级(如有)",
      "target": "适用人群(如有)",
      "sources": ["px"]
    }}
  ]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=4000)
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
        text = "\n---\n".join([c.text for c in chunks])[:12000]
        
        prompt = f"""从以下医学文献片段提取关键研究发现和结果。

文本：
{text}

要求：
1. 提取具体的研究结果和发现
2. 保留所有数值数据(p值、CI、效应量、百分比等)
3. 说明比较组别(如有)
4. 标注来源页码

返回JSON格式：
{{
  "key_findings": [
    {{
      "id": "F1",
      "finding": "具体发现内容",
      "data": "相关统计数据(p值/CI/效应量等)",
      "comparison": "比较组别(如有)",
      "sources": ["px"]
    }}
  ]
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=4000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                findings = data.get("key_findings", [])
                return deduplicate_findings(findings)
            except:
                pass
        return []
    
    def extract_conclusions(self, chunks: List[TextChunk]) -> Dict:
        """提取结论和局限性"""
        text = "\n---\n".join([c.text for c in chunks])[:10000]
        
        prompt = f"""从以下医学文献片段提取结论、局限性和未来方向。

文本：
{text}

返回JSON格式：
{{
  "conclusions": [
    {{
      "conclusion": "主要结论内容",
      "sources": ["px"]
    }}
  ],
  "limitations": "研究局限性(如有)",
  "future_directions": "未来研究方向(如有)",
  "clinical_implications": "临床意义(如有)"
}}

只返回JSON："""
        
        result = self._call_llm(prompt, max_tokens=2500)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                if "conclusions" in data:
                    data["conclusions"] = deduplicate_findings(data["conclusions"])
                return data
            except:
                pass
        return {"conclusions": []}


# ============ 主提取器 v2 ============
class FieldRetrievalExtractorV2:
    """方案5优化版: 字段级检索增强提取器"""
    
    def __init__(self, api_url: str = LOCAL_API, chunk_size: int = 5000, top_k: int = 6):
        self.api_url = api_url
        self.chunker = SmartPDFChunker(chunk_size=chunk_size)
        self.field_extractor = EnhancedFieldExtractor(api_url)
        self.top_k = top_k
    
    def classify(self, text: str) -> str:
        """分类文档类型"""
        prompt = """判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER
- GUIDELINE: 临床指南/诊疗规范/技术评估/实践建议
- REVIEW: 综述/系统评价/Meta分析
- OTHER: 原始研究/手册/报告/其他

文档开头：
""" + text[:4000] + """

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
            # Step 1: 智能分chunk
            chunks, chunk_stats = self.chunker.chunk_document(pdf_path)
            if not chunks:
                return {"success": False, "error": "PDF内容为空", "time": time.time() - start_time}
            
            total_pages = chunk_stats["total_doc_pages"]
            print(f"[方案5v2] 文档共{total_pages}页，保留{chunk_stats['kept_pages']}页，分成{len(chunks)}个chunk")
            print(f"[方案5v2] 跳过{chunk_stats['skipped_pages']}页(目录/参考文献等)")
            
            # Step 2: 分类
            first_chunk_text = chunks[0].text if chunks else ""
            doc_type = self.classify(first_chunk_text)
            print(f"[方案5v2] 文档类型: {doc_type}")
            
            # Step 3: 构建检索器
            retriever = EnhancedRetriever(chunks)
            
            # Step 4: 按字段检索并提取
            results = {"doc_type": doc_type}
            all_covered_pages: Set[int] = set()
            
            # 4.1 元数据
            metadata_chunks = retriever.retrieve("metadata", top_k=3)
            print(f"[方案5v2] 元数据检索: 页码 {sorted(set(p for c in metadata_chunks for p in c.pages))}")
            results["doc_metadata"] = self.field_extractor.extract_metadata(metadata_chunks)
            for c in metadata_chunks:
                all_covered_pages.update(c.pages)
            
            # 4.2 范围和目标
            scope_chunks = retriever.retrieve("scope", top_k=3)
            print(f"[方案5v2] 范围检索: 页码 {sorted(set(p for c in scope_chunks for p in c.pages))}")
            results["scope"] = self.field_extractor.extract_scope(scope_chunks)
            for c in scope_chunks:
                all_covered_pages.update(c.pages)
            
            # 4.3 推荐意见 (仅GUIDELINE)
            if doc_type == "GUIDELINE":
                rec_chunks = retriever.retrieve("recommendations", top_k=self.top_k)
                print(f"[方案5v2] 推荐检索: 页码 {sorted(set(p for c in rec_chunks for p in c.pages))}")
                results["recommendations"] = self.field_extractor.extract_recommendations(rec_chunks, doc_type)
                for c in rec_chunks:
                    all_covered_pages.update(c.pages)
            
            # 4.4 关键发现
            findings_chunks = retriever.retrieve("key_findings", top_k=self.top_k)
            print(f"[方案5v2] 发现检索: 页码 {sorted(set(p for c in findings_chunks for p in c.pages))}")
            results["key_findings"] = self.field_extractor.extract_findings(findings_chunks)
            for c in findings_chunks:
                all_covered_pages.update(c.pages)
            
            # 4.5 结论
            conclusions_chunks = retriever.retrieve("conclusions", top_k=self.top_k)
            print(f"[方案5v2] 结论检索: 页码 {sorted(set(p for c in conclusions_chunks for p in c.pages))}")
            conclusions_data = self.field_extractor.extract_conclusions(conclusions_chunks)
            results["conclusions"] = conclusions_data.get("conclusions", [])
            results["limitations"] = conclusions_data.get("limitations", "")
            results["future_directions"] = conclusions_data.get("future_directions", "")
            for c in conclusions_chunks:
                all_covered_pages.update(c.pages)
            
            return {
                "success": True,
                "doc_type": doc_type,
                "result": results,
                "time": time.time() - start_time,
                "stats": {
                    "total_pages": total_pages,
                    "kept_pages": chunk_stats["kept_pages"],
                    "skipped_pages": chunk_stats["skipped_pages"],
                    "total_chunks": len(chunks),
                    "covered_pages": sorted(all_covered_pages),
                    "coverage_ratio": len(all_covered_pages) / total_pages if total_pages > 0 else 0,
                    "findings_count": len(results.get("key_findings", [])),
                    "conclusions_count": len(results.get("conclusions", []))
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
    return FieldRetrievalExtractorV2().extract(pdf_path)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    
    print(f"\n{'='*50}")
    print(f"=== 方案5v2 测试结果 ===")
    print(f"{'='*50}")
    print(f"成功: {result.get('success')}")
    print(f"类型: {result.get('doc_type')}")
    print(f"耗时: {result.get('time', 0):.1f}s")
    
    if result.get("stats"):
        stats = result["stats"]
        print(f"\n--- 覆盖统计 ---")
        print(f"总页数: {stats.get('total_pages')}")
        print(f"保留页: {stats.get('kept_pages')} (跳过{stats.get('skipped_pages')}页)")
        print(f"覆盖页: {len(stats.get('covered_pages', []))}页")
        print(f"覆盖率: {stats.get('coverage_ratio', 0)*100:.1f}%")
        print(f"关键发现: {stats.get('findings_count')}条")
        print(f"结论: {stats.get('conclusions_count')}条")
    
    if result.get("success"):
        print(f"\n--- 提取结果预览 ---")
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:3000])
