"""
动态Few-shot提取器 - 基于文档类型加载对应的高质量样本作为few-shot示例
"""
import json
import os
import re
import fitz
import requests
from pathlib import Path

# API配置
LOCAL_API = "http://localhost:8000/v1/chat/completions"
FEWSHOT_DIR = Path("/root/pdf_summarization_deploy_20251225_093847/fewshot_samples")

def load_fewshot_sample(doc_type: str) -> dict:
    """加载对应类型的few-shot样本"""
    sample_path = FEWSHOT_DIR / f"{doc_type}_sample.json"
    if sample_path.exists():
        with open(sample_path) as f:
            return json.load(f)
    return None

def call_qwen(messages, max_tokens=8000):
    """调用Qwen API"""
    resp = requests.post(
        LOCAL_API,
        json={
            "model": "qwen3-8b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}
        },
        timeout=180
    )
    return resp.json()["choices"][0]["message"]["content"]

def extract_pdf_text(pdf_path: str, max_pages: int = 15) -> str:
    """提取PDF文本"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text()
        pages.append(f"=== 第{i+1}页 ===\n{text}")
    doc.close()
    return "\n".join(pages)

def classify_document(text: str) -> str:
    """文档分类"""
    prompt = """请判断这份医学文档的类型，只返回以下类型之一：
- GUIDELINE: 临床指南/实践指南/诊疗规范/技术评估（如NICE、WHO等权威机构发布的指导性文件）
- REVIEW: 综述/系统评价/Meta分析（学术研究类文献）
- OTHER: 其他类型（药品说明书、HTA报告、政策文件等）

文档开头：
""" + text[:3000] + """

只返回类型名称(GUIDELINE/REVIEW/OTHER)，不要其他内容。"""
    
    result = call_qwen([{"role": "user", "content": prompt}])
    result = result.strip().upper()
    for t in ["GUIDELINE", "REVIEW", "OTHER"]:
        if t in result:
            return t
    return "OTHER"

def build_extraction_prompt(doc_type: str, text: str, fewshot: dict) -> str:
    """构建带few-shot示例的提取prompt"""
    
    # 基础指令
    base_instruction = f"""你是专业医学文献信息提取专家。请从以下{doc_type}类型文档中提取结构化信息。

## 核心要求
1. **来源标注**: 每项信息必须用"sources"标注页码，格式如["p1"]或["p2-p3"]
2. **忠实原文**: 只提取文档中明确存在的信息，绝不编造
3. **完整提取**: 提取所有重要信息，尤其是推荐/建议/结论

"""
    
    # 添加few-shot示例
    fewshot_section = ""
    if fewshot:
        # 简化示例，只展示关键部分
        example_output = fewshot["expected_output"]
        simplified = {
            "doc_metadata": example_output.get("doc_metadata", {}),
        }
        # 根据类型添加关键字段
        if doc_type == "GUIDELINE":
            if "recommendations" in example_output:
                simplified["recommendations"] = example_output["recommendations"][:2]  # 只展示2个示例
        elif doc_type == "REVIEW":
            for k in ["scope", "key_findings", "conclusions"]:
                if k in example_output and example_output[k]:
                    simplified[k] = example_output[k][:2] if isinstance(example_output[k], list) else example_output[k]
        else:
            for k in ["scope", "key_findings"]:
                if k in example_output and example_output[k]:
                    simplified[k] = example_output[k][:2] if isinstance(example_output[k], list) else example_output[k]
        
        fewshot_section = f"""## 参考示例
以下是一个高质量提取的示例（注意来源标注格式）：
```json
{json.dumps(simplified, ensure_ascii=False, indent=2)[:2000]}
```

"""
    
    # 输出格式
    if doc_type == "GUIDELINE":
        output_format = """{
  "doc_metadata": {"title": "标题", "organization": "机构", "publish_date": "日期", "version": "版本", "sources": ["p1"]},
  "scope": {"population": "适用人群", "conditions": "适用情况", "sources": ["px"]},
  "recommendations": [
    {"id": "1.1", "text": "推荐内容", "strength": "强度", "evidence_level": "证据等级", "sources": ["px"]}
  ],
  "key_evidence": [{"id": "E1", "description": "关键证据", "sources": ["px"]}],
  "implementation": {"considerations": "实施要点", "sources": ["px"]}
}"""
    elif doc_type == "REVIEW":
        output_format = """{
  "doc_metadata": {"title": "标题", "authors": "作者", "journal": "期刊", "doi": "DOI", "sources": ["p1"]},
  "scope": {"objective": "研究目的", "sources": ["px"]},
  "methods": [{"id": "M1", "description": "方法描述", "sources": ["px"]}],
  "key_findings": [{"id": "F1", "category": "分类", "finding": "发现", "sources": ["px"]}],
  "conclusions": [{"id": "C1", "text": "结论", "sources": ["px"]}]
}"""
    else:
        output_format = """{
  "doc_metadata": {"title": "标题", "document_type": "文档类型", "organization": "机构", "sources": ["p1"]},
  "scope": {"description": "范围描述", "sources": ["px"]},
  "key_findings": [{"id": "F1", "category": "分类", "finding": "发现", "sources": ["px"]}],
  "conclusions": [{"id": "C1", "text": "结论", "sources": ["px"]}]
}"""
    
    # 最终prompt
    return f"""{base_instruction}{fewshot_section}## 输出格式
```json
{output_format}
```

## 待提取文档
{text}

请提取结构化信息，直接返回JSON。"""

def extract_with_fewshot(pdf_path: str) -> dict:
    """使用动态few-shot进行提取"""
    # 1. 提取文本
    text = extract_pdf_text(pdf_path)
    
    # 2. 分类
    doc_type = classify_document(text)
    
    # 3. 加载few-shot样本
    fewshot = load_fewshot_sample(doc_type)
    
    # 4. 构建prompt并提取
    prompt = build_extraction_prompt(doc_type, text, fewshot)
    result = call_qwen([{"role": "user", "content": prompt}])
    
    # 5. 解析结果
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "success": True,
                "doc_type": doc_type,
                "has_fewshot": fewshot is not None,
                "result": data
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON解析失败: {e}", "raw": result[:500]}
    return {"success": False, "error": "未找到JSON", "raw": result[:500]}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf = sys.argv[1]
    else:
        pdf = "/root/autodl-tmp/pdf_summarization_data/pdf_samples/1756864558155_1608702.pdf"
    
    print(f"测试PDF: {pdf}")
    result = extract_with_fewshot(pdf)
    print(f"分类: {result.get('doc_type', 'N/A')}")
    print(f"Few-shot: {result.get('has_fewshot', False)}")
    print(f"成功: {result.get('success', False)}")
    if result.get("success"):
        print(json.dumps(result["result"], ensure_ascii=False, indent=2)[:1000])
