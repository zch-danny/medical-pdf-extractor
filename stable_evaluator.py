#!/usr/bin/env python3
"""
稳定版GPT评分器
特点:
1. 详细的评分rubric (每个分数段有明确标准)
2. Few-shot示例锚定评分标准
3. 多次评分取平均+标准差
4. 基于引用页码评估
"""

import os, sys, json, re, time
import fitz
import requests
from typing import Dict, List, Tuple, Optional

API_KEY = os.environ.get('GPT_API_KEY', 'sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7')
API_DOMAIN = os.environ.get('GPT_API_DOMAIN', 'api.bltcy.ai')

# 详细评分标准 (rubric)
EVAL_PROMPT_TEMPLATE = '''你是医学文献信息提取质量评估专家。请严格按照以下标准评分。

## 评分标准 (Rubric)

### 1. accuracy (准确性) - 提取内容是否在原文中真实存在
- 9-10分: 所有提取内容都能在原文中找到原句或近似表述
- 7-8分: 大部分内容准确，少量概括性表述但无明显错误
- 5-6分: 部分内容准确，存在一些推断或不精确表述
- 3-4分: 多处内容无法在原文中验证，存在明显错误
- 1-2分: 大量编造或严重失实

### 2. completeness (完整性) - 关键医学信息是否提取完整
- 9-10分: 提取了原文中所有关键推荐/发现/证据
- 7-8分: 提取了大部分关键信息，遗漏少量次要内容
- 5-6分: 提取了主要信息，但遗漏部分重要内容
- 3-4分: 仅提取了少量关键信息，遗漏较多
- 1-2分: 几乎没有提取到关键信息

### 3. source_accuracy (来源准确性) - 页码标注是否正确
- 9-10分: 所有页码标注都准确对应原文位置
- 7-8分: 大部分页码准确，少量偏差在±1页内
- 5-6分: 部分页码准确，存在较明显偏差
- 3-4分: 多数页码不准确或无法验证
- 1-2分: 页码标注基本无效

## Few-shot评分示例

示例1 (高质量提取, 应得8-9分):
- 提取内容几乎都能在原文找到对应句子
- 推荐内容完整，包含强度和证据等级
- 页码标注准确
→ accuracy=9, completeness=8, source_accuracy=9, overall=8.7

示例2 (中等质量, 应得5-6分):
- 部分内容准确，但有概括性推断
- 遗漏了一些重要推荐
- 页码有偏差
→ accuracy=6, completeness=5, source_accuracy=5, overall=5.3

示例3 (低质量, 应得2-3分):
- 多处内容无法在原文验证
- 关键信息严重遗漏
- 页码标注错误
→ accuracy=3, completeness=2, source_accuracy=2, overall=2.3

## 待评估内容

### 原文 (参考页面: {ref_pages}):
{pdf_text}

### AI提取结果:
{extraction}

## 请评分

请严格按照上述rubric评分，返回JSON格式:
{{"accuracy": 分数, "completeness": 分数, "source_accuracy": 分数, "overall": 三项平均值, "reason": "一句话说明主要扣分原因"}}

只返回JSON，不要其他内容:'''


PAGE_RE = re.compile(r'p(\d+)(?:\s*-\s*p?(\d+))?', re.I)

def parse_referenced_pages(result_dict: Dict, total_pages: int) -> List[int]:
    """从提取结果中解析引用的页码"""
    pages = set([1, 2, 3])  # 始终包含前3页
    if total_pages > 5:
        pages.update([total_pages-1, total_pages])  # 最后2页
    
    data = result_dict or {}
    for field in ['recommendations', 'key_findings', 'key_evidence']:
        for item in (data.get(field) or []):
            if not isinstance(item, dict):
                continue
            for src in (item.get('sources') or []):
                for m in PAGE_RE.finditer(str(src)):
                    a, b = m.group(1), m.group(2)
                    if a and not b:
                        pages.add(int(a))
                    elif a and b:
                        for x in range(int(a), int(b)+1):
                            pages.add(x)
    
    return sorted(p for p in pages if 1 <= p <= total_pages)

def extract_pages_text(pdf_path: str, page_numbers: List[int], max_chars: int = 15000) -> str:
    """提取指定页码的文本"""
    doc = fitz.open(pdf_path)
    texts = []
    for p in page_numbers:
        if 1 <= p <= len(doc):
            txt = doc[p-1].get_text()[:2500]
            texts.append(f"=== 第{p}页 ===\n{txt}")
    doc.close()
    return "\n".join(texts)[:max_chars]

def call_gpt(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    """调用GPT API，temperature=0确保确定性"""
    url = f"https://{API_DOMAIN}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,  # 0=完全确定性
        "seed": 42  # 固定seed进一步提高稳定性
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    result = resp.json()
    if "error" in result:
        raise Exception(f"API错误: {result}")
    return result["choices"][0]["message"]["content"]

def evaluate_once(pdf_text: str, extraction: Dict, ref_pages: List[int]) -> Dict:
    """单次评分"""
    prompt = EVAL_PROMPT_TEMPLATE.format(
        ref_pages=ref_pages[:15],
        pdf_text=pdf_text[:12000],
        extraction=json.dumps(extraction, ensure_ascii=False, indent=2)[:5000]
    )
    
    resp = call_gpt(prompt, temperature=0.0)
    match = re.search(r'\{[^}]+\}', resp.replace('\n', ' '))
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"overall": 0, "error": "解析失败"}

def evaluate_stable(pdf_path: str, extraction: Dict, n_runs: int = 3) -> Dict:
    """稳定评分: 多次评分取平均"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    
    ref_pages = parse_referenced_pages(extraction, total_pages)
    pdf_text = extract_pages_text(pdf_path, ref_pages)
    
    scores = []
    for i in range(n_runs):
        score = evaluate_once(pdf_text, extraction, ref_pages)
        if score.get("overall", 0) > 0:
            scores.append(score)
        time.sleep(0.5)  # 避免rate limit
    
    if not scores:
        return {"overall": 0, "error": "所有评分失败", "n_runs": n_runs}
    
    # 计算平均和标准差
    avg_overall = sum(s.get("overall", 0) for s in scores) / len(scores)
    avg_acc = sum(s.get("accuracy", 0) for s in scores) / len(scores)
    avg_comp = sum(s.get("completeness", 0) for s in scores) / len(scores)
    avg_src = sum(s.get("source_accuracy", 0) for s in scores) / len(scores)
    
    # 标准差
    if len(scores) > 1:
        variance = sum((s.get("overall", 0) - avg_overall) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
    else:
        std = 0
    
    return {
        "overall": round(avg_overall, 2),
        "accuracy": round(avg_acc, 2),
        "completeness": round(avg_comp, 2),
        "source_accuracy": round(avg_src, 2),
        "std": round(std, 2),
        "n_runs": len(scores),
        "ref_pages": ref_pages,
        "details": scores
    }

if __name__ == "__main__":
    # 测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="PDF文件路径")
    parser.add_argument("--extraction", help="提取结果JSON文件")
    parser.add_argument("--runs", type=int, default=3, help="评分次数")
    args = parser.parse_args()
    
    if args.extraction:
        with open(args.extraction, 'r') as f:
            extraction = json.load(f)
    else:
        # 使用v7.11提取
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from production_extractor_v711_quote import MedicalPDFExtractorV711
        result = MedicalPDFExtractorV711().extract(args.pdf)
        extraction = result.get("result", {})
    
    score = evaluate_stable(args.pdf, extraction, args.runs)
    print(json.dumps(score, ensure_ascii=False, indent=2))
