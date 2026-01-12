#!/usr/bin/env python3
"""
三阶段医学文献提取器
阶段1: 分类 + 元数据提取 (前2页)
阶段2: 逐页内容提取 (每页单独处理)
阶段3: 汇总整合
"""
import json, time, requests, fitz, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 配置
QWEN_API = "http://localhost:8000/v1/chat/completions"
MAX_PAGES = 5
CHARS_PER_PAGE = 2000

def call_qwen(prompt: str, max_tokens: int = 2000) -> Tuple[str, float, int]:
    """调用Qwen API"""
    start = time.time()
    r = requests.post(QWEN_API, json={
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=120)
    d = r.json()
    content = d['choices'][0]['message']['content']
    tokens = d.get('usage', {}).get('total_tokens', 0)
    return content, time.time() - start, tokens

def parse_json(text: str) -> Optional[Dict]:
    """解析JSON"""
    text = re.sub(r'```json\s*|\s*```', '', text)
    m = re.search(r'\{[\s\S]+\}', text)
    if m:
        try:
            return json.loads(m.group())
        except:
            return None
    return None

def extract_pages(pdf_path: str) -> List[str]:
    """提取每页文本"""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(min(MAX_PAGES, len(doc))):
        text = doc[i].get_text()[:CHARS_PER_PAGE]
        pages.append(f"[第{i+1}页]\n{text}")
    doc.close()
    return pages

# ============ 阶段1: 分类 + 元数据 ============
STAGE1_PROMPT = """# 阶段1：分类与元数据提取

从以下文献前2页提取基本信息。

## 输入
{text}

## 任务
1. 判断文献类型: GUIDELINE/META/REVIEW/RCT/COHORT/OTHER
2. 提取元数据: 标题、作者、机构、年份、DOI等

## 输出JSON
```json
{{
  "type": "文献类型",
  "confidence": "high/medium/low",
  "metadata": {{
    "title": "",
    "authors": "",
    "organization": "",
    "year": "",
    "doi": "",
    "document_type": "",
    "sources": ["页码"]
  }}
}}
```
只输出JSON。"""

def stage1_classify_and_metadata(pages: List[str]) -> Dict:
    """阶段1: 分类和元数据提取"""
    text = "\n".join(pages[:2])  # 只用前2页
    prompt = STAGE1_PROMPT.replace("{text}", text)
    content, elapsed, tokens = call_qwen(prompt, 1500)
    result = parse_json(content)
    return {
        "result": result,
        "time": elapsed,
        "tokens": tokens
    }

# ============ 阶段2: 逐页内容提取 ============
STAGE2_PROMPT_GUIDELINE = """# 阶段2：单页内容提取 (指南类)

从第{page_num}页提取关键信息。

## 当前页内容
{text}

## 任务
提取该页中的:
1. 推荐意见 (recommendations) - 包含should/recommend/建议的句子
2. 关键发现 (findings)
3. 重要数据点

## 输出JSON
```json
{{
  "page": "p{page_num}",
  "recommendations": [
    {{"id": "编号", "text": "原文内容", "source": "p{page_num}"}}
  ],
  "findings": [
    {{"text": "发现内容", "source": "p{page_num}"}}
  ],
  "data_points": [
    {{"description": "", "value": "", "source": "p{page_num}"}}
  ]
}}
```
只输出JSON。如果该页没有相关内容，对应数组为空。"""

STAGE2_PROMPT_OTHER = """# 阶段2：单页内容提取 (通用)

从第{page_num}页提取关键信息。

## 当前页内容
{text}

## 任务
提取该页中的:
1. 主要观点/结论
2. 关键数据
3. 重要引用

## 输出JSON
```json
{{
  "page": "p{page_num}",
  "main_points": ["观点1", "观点2"],
  "data": [{{"description": "", "value": "", "source": "p{page_num}"}}],
  "quotes": [{{"text": "", "source": "p{page_num}"}}]
}}
```
只输出JSON。"""

def stage2_extract_page(page_text: str, page_num: int, doc_type: str) -> Dict:
    """阶段2: 单页提取"""
    if doc_type in ["GUIDELINE", "META", "RCT"]:
        prompt_template = STAGE2_PROMPT_GUIDELINE
    else:
        prompt_template = STAGE2_PROMPT_OTHER
    
    prompt = prompt_template.replace("{text}", page_text).replace("{page_num}", str(page_num))
    content, elapsed, tokens = call_qwen(prompt, 1500)
    result = parse_json(content)
    return {
        "page": page_num,
        "result": result,
        "time": elapsed,
        "tokens": tokens
    }

# ============ 阶段3: 汇总整合 ============
STAGE3_PROMPT = """# 阶段3：汇总整合

将各页提取结果整合为最终输出。

## 元数据
{metadata}

## 各页提取结果
{page_results}

## 任务
1. 合并所有recommendations，去重
2. 合并所有findings
3. 保留准确的页码标注
4. 生成完整的结构化输出

## 输出JSON
```json
{{
  "doc_metadata": {{
    "title": "",
    "authors": "",
    "organization": "",
    "year": "",
    "doi": "",
    "document_type": "",
    "sources": []
  }},
  "recommendations": [
    {{"id": "", "text": "", "sources": []}}
  ],
  "key_findings": [
    {{"finding": "", "sources": []}}
  ],
  "extraction_quality": {{
    "pages_processed": [],
    "completeness": "High/Medium/Low"
  }}
}}
```
只输出JSON。"""

def stage3_merge(metadata: Dict, page_results: List[Dict]) -> Dict:
    """阶段3: 汇总整合"""
    metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)
    pages_str = json.dumps(page_results, ensure_ascii=False, indent=2)
    
    prompt = STAGE3_PROMPT.replace("{metadata}", metadata_str).replace("{page_results}", pages_str)
    content, elapsed, tokens = call_qwen(prompt, 3000)
    result = parse_json(content)
    return {
        "result": result,
        "time": elapsed,
        "tokens": tokens
    }

# ============ 主流程 ============
def three_stage_extract(pdf_path: str) -> Dict:
    """三阶段提取主流程"""
    print(f"处理: {Path(pdf_path).name}")
    total_start = time.time()
    total_tokens = 0
    
    # 提取页面
    pages = extract_pages(pdf_path)
    print(f"  共{len(pages)}页")
    
    # 阶段1: 分类 + 元数据
    print("  [阶段1] 分类+元数据...", end=" ")
    stage1 = stage1_classify_and_metadata(pages)
    total_tokens += stage1['tokens']
    
    if not stage1['result']:
        return {"error": "阶段1失败", "stage1": stage1}
    
    doc_type = stage1['result'].get('type', 'OTHER')
    metadata = stage1['result'].get('metadata', {})
    print(f"类型={doc_type}, {stage1['time']:.1f}s")
    
    # 阶段2: 逐页提取
    print("  [阶段2] 逐页提取...")
    page_results = []
    for i, page_text in enumerate(pages):
        page_num = i + 1
        print(f"    第{page_num}页...", end=" ")
        page_result = stage2_extract_page(page_text, page_num, doc_type)
        total_tokens += page_result['tokens']
        page_results.append(page_result['result'])
        print(f"{page_result['time']:.1f}s")
    
    # 阶段3: 汇总
    print("  [阶段3] 汇总整合...", end=" ")
    stage3 = stage3_merge(metadata, page_results)
    total_tokens += stage3['tokens']
    print(f"{stage3['time']:.1f}s")
    
    total_time = time.time() - total_start
    
    return {
        "pdf": Path(pdf_path).name,
        "doc_type": doc_type,
        "final_result": stage3['result'],
        "stages": {
            "stage1": stage1,
            "stage2": page_results,
            "stage3": stage3
        },
        "performance": {
            "total_time": total_time,
            "total_tokens": total_tokens,
            "api_calls": 1 + len(pages) + 1  # stage1 + pages + stage3
        }
    }

def main():
    """测试入口"""
    import sys
    
    # 默认测试文件
    pdf_dir = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
    pdfs = list(pdf_dir.glob('*.pdf'))[:3]  # 先测3个
    
    print("="*60)
    print("三阶段提取器测试")
    print("="*60)
    
    results = []
    for pdf in pdfs:
        result = three_stage_extract(str(pdf))
        results.append(result)
        print(f"  总耗时: {result['performance']['total_time']:.1f}s")
        print(f"  API调用: {result['performance']['api_calls']}次")
        print(f"  总tokens: {result['performance']['total_tokens']}")
        print()
    
    # 保存结果
    output_dir = Path('/root/extraction_test_results/three_stage_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{ts}.json", 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存: {output_dir}")
    
    # 汇总
    times = [r['performance']['total_time'] for r in results]
    tokens = [r['performance']['total_tokens'] for r in results]
    print(f"\n平均耗时: {sum(times)/len(times):.1f}s")
    print(f"平均tokens: {sum(tokens)//len(tokens)}")

if __name__ == '__main__':
    main()
