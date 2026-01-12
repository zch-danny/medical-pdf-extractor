"""
方案5优化版 v3: 字段级检索增强提取器
新增优化:
1. 并行提取各字段(3-4x加速)
2. 增加top_k和overlap提高覆盖率
3. 二次检索：首次低分chunk补充检索
4. 输出格式与原v7兼容
5. 更精细的去重和质量控制
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

# ============ 数据结构 ============
@dataclass
class TextChunk:
    index: int
    text: str
    pages: List[int]
    char_count: int

# ============ 工具函数 ============
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

def deduplicate_items(items: List[Dict], key_field: str = "finding", threshold: float = 0.6) -> List[Dict]:
    """去重：基于文本相似度"""
    if not items:
        return []
    
    result = []
    seen_texts = []
    
    for item in items:
        text = item.get(key_field, item.get("conclusion", item.get("text", "")))
        if not text or len(text) < 10:
            continue
        
        is_dup = False
        text_words = set(text.lower().split())
        
        for seen in seen_texts:
            seen_words = set(seen.split())
            if len(text_words) > 0 and len(seen_words) > 0:
                overlap = len(text_words & seen_words) / min(len(text_words), len(seen_words))
                if overlap > threshold:
                    is_dup = True
                    break
        
        if not is_dup:
            result.append(item)
            seen_texts.append(text.lower())
    
    return result

def normalize_sources(sources: Any) -> List[str]:
    """标准化sources格式"""
    if not sources:
        return []
    if isinstance(sources, str):
        return [sources]
    if isinstance(sources, list):
        result = []
        for s in sources:
            if isinstance(s, str):
                result.append(s)
            elif isinstance(s, int):
                result.append(f"p{s}")
        return result
    return []


# ============ 智能分块器 v3 ============
class SmartPDFChunkerV3:
    """智能PDF分块器：更大overlap，更精细的跳页逻辑"""
    
    SKIP_PATTERNS = [
        r'^table\s+of\s+contents?$',
        r'^contents?$',
        r'^目录$',
        r'^references?$',
        r'^bibliography$',
        r'^参考文献$',
        r'^appendix\s*[a-z0-9]*$',
        r'^附录',
        r'^index$',
        r'^acknowledgement',
        r'^致谢',
        r'^abbreviations?$',
        r'^缩略语',
        r'^glossary$',
        r'^术语表',
    ]
    
    def __init__(self, chunk_size: int = 4500, overlap: int = 600):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._skip_re = [re.compile(p, re.IGNORECASE) for p in self.SKIP_PATTERNS]
    
    def _should_skip_page(self, text: str, page_num: int, total_pages: int) -> bool:
        if len(text.strip()) < 80:
            return True
        
        first_lines = '\n'.join(text.strip().split('\n')[:3])
        for pattern in self._skip_re:
            if pattern.search(first_lines):
                return True
        
        # 参考文献特征检测
        ref_pattern = r'\(\d{4}\)|\d{4}[;,]|\[\d+\]'
        ref_count = len(re.findall(ref_pattern, text))
        if ref_count > 15 and page_num > total_pages * 0.7:
            return True
        
        return False
    
    def _is_key_page(self, text: str, page_num: int, total_pages: int) -> bool:
        """判断是否为关键页"""
        text_lower = text.lower()[:1000]
        
        # 首尾页
        if page_num <= 2 or page_num >= total_pages - 1:
            return True
        
        # 包含关键章节标题
        key_sections = ['abstract', 'summary', 'conclusion', 'result', 'finding',
                        'recommendation', 'discussion', '摘要', '结论', '结果', '推荐']
        for kw in key_sections:
            if kw in text_lower:
                return True
        
        return False
    
    def chunk_document(self, pdf_path: str) -> Tuple[List[TextChunk], Dict]:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        
        # 提取有效页面
        valid_pages = []
        skipped_count = 0
        key_page_nums = []
        
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            page_num = i + 1
            
            if self._should_skip_page(text, page_num, total_pages):
                skipped_count += 1
                continue
            
            is_key = self._is_key_page(text, page_num, total_pages)
            if is_key:
                key_page_nums.append(page_num)
            
            valid_pages.append((page_num, text))
        
        doc.close()
        
        if not valid_pages:
            return [], {"total_doc_pages": total_pages, "kept_pages": 0, "skipped_pages": skipped_count}
        
        # 合并文本
        full_text = ""
        page_positions = []
        
        for page_num, text in valid_pages:
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
                for sep in ['\n\n', '\n', '. ', '。', '; ', '；']:
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
            "total_doc_pages": total_pages,
            "kept_pages": len(valid_pages),
            "skipped_pages": skipped_count,
            "key_pages": key_page_nums,
            "total_chunks": len(chunks)
        }
        
        return chunks, stats


# ============ 增强检索器 v3 ============
class EnhancedRetrieverV3:
    """增强版检索器：二次检索补充"""
    
    FIELD_KEYWORDS = {
        "metadata": [
            "title", "author", "abstract", "introduction", "doi", "journal",
            "published", "copyright", "correspondence", "affiliation",
            "background", "objective", "purpose", "aim", "issn", "isbn",
            "标题", "作者", "摘要", "引言", "期刊", "背景", "目的"
        ],
        "scope": [
            "scope", "objective", "aim", "purpose", "target", "population",
            "inclusion", "exclusion", "criteria", "setting", "context",
            "methods", "methodology", "study design", "approach", "protocol",
            "范围", "目标", "纳入", "排除", "标准", "方法", "设计"
        ],
        "recommendations": [
            "recommend", "recommendation", "should", "must", "suggest",
            "advise", "guideline", "guidance", "statement", "consensus",
            "strong recommendation", "weak recommendation", "conditional",
            "grade", "level of evidence", "quality of evidence", "GRADE",
            "class i", "class ii", "class iii", "level a", "level b", "level c",
            "good practice", "expert opinion", "clinical judgment",
            "推荐", "建议", "应该", "必须", "指南", "强推荐", "弱推荐", "共识"
        ],
        "key_findings": [
            "result", "finding", "found", "showed", "demonstrated", "revealed",
            "evidence", "significant", "outcome", "effect", "efficacy", "safety",
            "compared", "versus", "trial", "study", "analysis", "data",
            "p-value", "p value", "p<", "p=", "p =", "ci", "confidence interval",
            "hazard ratio", "odds ratio", "risk ratio", "relative risk", "rr",
            "absolute risk", "ard", "number needed", "nnt", "nntt", "or", "hr",
            "mean difference", "md", "smd", "standardized mean", "forest plot",
            "heterogeneity", "i2", "i²", "meta-analysis", "pooled",
            "sensitivity", "specificity", "auc", "roc",
            "95%", "99%", "0.05", "0.01", "0.001", "statistically",
            "结果", "发现", "显示", "证据", "显著", "效果", "安全性",
            "风险比", "优势比", "置信区间", "异质性", "敏感性", "特异性"
        ],
        "conclusions": [
            "conclusion", "conclusions", "summary", "in summary", "overall",
            "in conclusion", "we conclude", "this study", "our findings",
            "limitation", "limitations", "weakness", "strength", "strengths",
            "implication", "implications", "future", "further research",
            "clinical significance", "practical implications", "policy",
            "take-away", "key message", "main finding",
            "结论", "总结", "综上所述", "局限", "局限性", "启示", "意义",
            "未来研究", "临床意义", "关键信息"
        ]
    }
    
    def __init__(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self._build_index()
    
    def _build_index(self):
        self.doc_freq = Counter()
        self.term_freq = []
        self.chunk_scores = {}  # 缓存各字段的分数
        
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
    
    def _score_chunk(self, chunk_idx: int, field: str) -> float:
        keywords = self.FIELD_KEYWORDS.get(field, [])
        if not keywords:
            return 0.0
        
        score = 0.0
        tf = self.term_freq[chunk_idx]
        n_docs = len(self.chunks)
        
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in tf:
                term_f = tf[kw_lower]
                doc_f = self.doc_freq.get(kw_lower, 1)
                idf = math.log((n_docs - doc_f + 0.5) / (doc_f + 0.5) + 1)
                score += term_f * idf
        
        # 位置加权
        if chunk_idx == 0:
            score *= 1.4
        elif chunk_idx == n_docs - 1:
            score *= 1.25
        elif chunk_idx <= 2:
            score *= 1.15
        
        return score
    
    def retrieve(self, field: str, top_k: int = 8) -> List[TextChunk]:
        """检索相关chunk，支持二次检索补充"""
        scores = [(self._score_chunk(i, field), i, c) for i, c in enumerate(self.chunks)]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        seen_indices = set()
        
        # 主检索
        for score, idx, chunk in scores:
            if score > 0.5 and idx not in seen_indices and len(result) < top_k:
                result.append(chunk)
                seen_indices.add(idx)
        
        # 二次检索：如果结果不足，降低阈值
        if len(result) < top_k // 2:
            for score, idx, chunk in scores:
                if score > 0 and idx not in seen_indices and len(result) < top_k:
                    result.append(chunk)
                    seen_indices.add(idx)
        
        # 确保包含首尾
        n_docs = len(self.chunks)
        if 0 not in seen_indices:
            result.insert(0, self.chunks[0])
        if n_docs > 1 and (n_docs - 1) not in seen_indices:
            result.append(self.chunks[-1])
        
        return result if result else [self.chunks[0]]


# ============ 并行字段提取器 ============
class ParallelFieldExtractor:
    """并行字段提取器：加速多字段提取"""
    
    def __init__(self, api_url: str = LOCAL_API, max_workers: int = 3):
        self.api_url = api_url
        self.max_workers = max_workers
    
    def _call_llm(self, prompt: str, max_tokens: int = 2500) -> str:
        resp = requests.post(
            self.api_url,
            json={
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.12,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            timeout=200
        )
        data = resp.json()
        if "choices" not in data:
            raise Exception(f"LLM API错误: {data.get('error', data)}")
        return data["choices"][0]["message"]["content"]
    
    def extract_metadata(self, chunks: List[TextChunk]) -> Dict:
        text = "\n---\n".join([c.text for c in chunks])[:12000]
        prompt = f"""从以下医学文献片段提取基本信息。

文本：
{text}

返回JSON：
{{
  "title": "完整标题",
  "authors": "作者(逗号分隔)",
  "organization": "机构/期刊",
  "publish_date": "年份",
  "document_type": "类型(指南/综述/研究/手册等)",
  "doi": "DOI(如有)",
  "language": "语言",
  "sources": ["p1"]
}}
只返回JSON："""
        
        result = self._call_llm(prompt, 1000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                data["sources"] = normalize_sources(data.get("sources"))
                return data
            except:
                pass
        return {"title": "Unknown", "sources": []}
    
    def extract_scope(self, chunks: List[TextChunk]) -> Dict:
        text = "\n---\n".join([c.text for c in chunks])[:10000]
        prompt = f"""从以下医学文献提取研究范围和方法。

文本：
{text}

返回JSON：
{{
  "content_summary": "内容概述(2-3句)",
  "objectives": "研究目标",
  "target_population": "目标人群(如有)",
  "setting": "场景/环境(如有)",
  "methods_summary": "方法概述(如有)",
  "sources": ["px"]
}}
只返回JSON："""
        
        result = self._call_llm(prompt, 1200)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                data["sources"] = normalize_sources(data.get("sources"))
                return data
            except:
                pass
        return {"content_summary": "", "sources": []}
    
    def extract_recommendations(self, chunks: List[TextChunk]) -> List[Dict]:
        text = "\n---\n".join([c.text for c in chunks])[:14000]
        prompt = f"""从以下临床指南提取所有推荐意见。

文本：
{text}

要求：
1. 提取具体临床推荐，不是章节标题
2. 保留推荐强度和证据等级
3. 完整引用原文
4. 标注页码

返回JSON：
{{
  "recommendations": [
    {{
      "id": "1.1",
      "text": "完整推荐内容",
      "strength": "推荐强度",
      "evidence_level": "证据等级",
      "sources": ["px"]
    }}
  ]
}}
只返回JSON："""
        
        result = self._call_llm(prompt, 4500)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                recs = data.get("recommendations", [])
                for r in recs:
                    r["sources"] = normalize_sources(r.get("sources"))
                return recs
            except:
                pass
        return []
    
    def extract_findings(self, chunks: List[TextChunk]) -> List[Dict]:
        text = "\n---\n".join([c.text for c in chunks])[:14000]
        prompt = f"""从以下医学文献提取关键研究发现。

文本：
{text}

要求：
1. 提取具体研究结果
2. 保留数值(p值/CI/效应量/百分比)
3. 标注页码

返回JSON：
{{
  "key_findings": [
    {{
      "finding": "具体发现",
      "data": "统计数据(如有)",
      "sources": ["px"]
    }}
  ]
}}
只返回JSON："""
        
        result = self._call_llm(prompt, 4000)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                findings = data.get("key_findings", [])
                for f in findings:
                    f["sources"] = normalize_sources(f.get("sources"))
                return deduplicate_items(findings, "finding")
            except:
                pass
        return []
    
    def extract_conclusions(self, chunks: List[TextChunk]) -> Dict:
        text = "\n---\n".join([c.text for c in chunks])[:12000]
        prompt = f"""从以下医学文献提取结论和局限性。

文本：
{text}

返回JSON：
{{
  "conclusions": [
    {{
      "conclusion": "主要结论",
      "sources": ["px"]
    }}
  ],
  "limitations": "局限性(如有)",
  "implications": "临床/实践意义(如有)"
}}
只返回JSON："""
        
        result = self._call_llm(prompt, 2500)
        json_str = extract_first_json(result)
        if json_str:
            try:
                data = json.loads(json_str)
                if "conclusions" in data:
                    for c in data["conclusions"]:
                        c["sources"] = normalize_sources(c.get("sources"))
                    data["conclusions"] = deduplicate_items(data["conclusions"], "conclusion")
                return data
            except:
                pass
        return {"conclusions": []}
    
    def extract_parallel(self, retriever, doc_type: str, top_k: int = 8) -> Dict:
        """并行提取所有字段"""
        results = {}
        
        # 准备各字段的chunks
        tasks = {
            "metadata": (self.extract_metadata, retriever.retrieve("metadata", top_k=4)),
            "scope": (self.extract_scope, retriever.retrieve("scope", top_k=4)),
            "key_findings": (self.extract_findings, retriever.retrieve("key_findings", top_k=top_k)),
            "conclusions": (self.extract_conclusions, retriever.retrieve("conclusions", top_k=top_k)),
        }
        
        if doc_type == "GUIDELINE":
            tasks["recommendations"] = (self.extract_recommendations, retriever.retrieve("recommendations", top_k=top_k))
        
        all_covered_pages = set()
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for field, (func, chunks) in tasks.items():
                for c in chunks:
                    all_covered_pages.update(c.pages)
                futures[executor.submit(func, chunks)] = field
            
            for future in as_completed(futures):
                field = futures[future]
                try:
                    result = future.result()
                    if field == "conclusions":
                        results["conclusions"] = result.get("conclusions", [])
                        results["limitations"] = result.get("limitations", "")
                        results["implications"] = result.get("implications", "")
                    else:
                        results[field] = result
                except Exception as e:
                    print(f"[警告] {field}提取失败: {e}")
                    results[field] = [] if field in ["recommendations", "key_findings"] else {}
        
        return results, all_covered_pages


# ============ 主提取器 v3 ============
class FieldRetrievalExtractorV3:
    """方案5 v3: 并行增强版"""
    
    def __init__(self, api_url: str = LOCAL_API, chunk_size: int = 4500, top_k: int = 8):
        self.api_url = api_url
        self.chunker = SmartPDFChunkerV3(chunk_size=chunk_size, overlap=600)
        self.field_extractor = ParallelFieldExtractor(api_url, max_workers=3)
        self.top_k = top_k
    
    def classify(self, text: str) -> str:
        prompt = """判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER
- GUIDELINE: 临床指南/诊疗规范/共识/实践建议
- REVIEW: 综述/系统评价/Meta分析
- OTHER: 原始研究/手册/报告/其他

文档开头：
""" + text[:5000] + """

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
        start_time = time.time()
        
        try:
            # Step 1: 分chunk
            chunks, chunk_stats = self.chunker.chunk_document(pdf_path)
            if not chunks:
                return {"success": False, "error": "PDF内容为空", "time": time.time() - start_time}
            
            total_pages = chunk_stats["total_doc_pages"]
            print(f"[v3] 文档{total_pages}页，保留{chunk_stats['kept_pages']}页，{len(chunks)}个chunk")
            
            # Step 2: 分类
            doc_type = self.classify(chunks[0].text)
            print(f"[v3] 类型: {doc_type}")
            
            # Step 3: 构建检索器
            retriever = EnhancedRetrieverV3(chunks)
            
            # Step 4: 并行提取
            print(f"[v3] 开始并行提取...")
            results, covered_pages = self.field_extractor.extract_parallel(retriever, doc_type, self.top_k)
            results["doc_type"] = doc_type
            
            # 确保格式兼容v7
            if "doc_metadata" not in results:
                results["doc_metadata"] = results.pop("metadata", {})
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "doc_type": doc_type,
                "result": results,
                "time": elapsed,
                "stats": {
                    "total_pages": total_pages,
                    "kept_pages": chunk_stats["kept_pages"],
                    "skipped_pages": chunk_stats["skipped_pages"],
                    "total_chunks": len(chunks),
                    "covered_pages": sorted(covered_pages),
                    "coverage_ratio": len(covered_pages) / total_pages if total_pages > 0 else 0,
                    "findings_count": len(results.get("key_findings", [])),
                    "conclusions_count": len(results.get("conclusions", [])),
                    "recommendations_count": len(results.get("recommendations", []))
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
    return FieldRetrievalExtractorV3().extract(pdf_path)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    result = extract_pdf(pdf)
    
    print(f"\n{'='*60}")
    print(f"方案5 v3 测试结果")
    print(f"{'='*60}")
    print(f"成功: {result.get('success')}")
    print(f"类型: {result.get('doc_type')}")
    print(f"耗时: {result.get('time', 0):.1f}s")
    
    if result.get("stats"):
        s = result["stats"]
        print(f"\n--- 统计 ---")
        print(f"总页: {s['total_pages']} | 保留: {s['kept_pages']} | 跳过: {s['skipped_pages']}")
        print(f"覆盖: {len(s['covered_pages'])}页 ({s['coverage_ratio']*100:.1f}%)")
        print(f"发现: {s['findings_count']} | 结论: {s['conclusions_count']} | 推荐: {s['recommendations_count']}")
    
    if result.get("success"):
        print(f"\n--- 结果预览 ---")
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:3500])
