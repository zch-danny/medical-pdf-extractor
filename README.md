# Medical PDF Extractor

基于 Qwen3-8B 的医学文献结构化信息提取系统。

## 功能特点

- **自动分类**: 识别文档类型 (GUIDELINE/REVIEW/OTHER)
- **动态Few-shot**: 根据类型加载对应高质量示例
- **结构化提取**: 提取元数据、推荐意见、关键发现、结论等
- **来源追溯**: 每项信息标注页码来源

## 性能指标

### v7.2 测试结果 (100个PDF)

| 指标 | 结果 |
|------|------|
| 成功率 | 97% |
| 平均评分 | 9.60/10 (GPT-5.2评估) |
| ≥8分占比 | 100% |
| 平均耗时 | ~58秒/文件 |

### 各维度评分
- accuracy (准确性): 9.0+
- completeness (完整性): 8.0+
- structure (结构性): 9.0+
- page_accuracy (页码准确): 8.5+
- no_hallucination (无编造): 9.5+

## 快速开始

### 环境要求
- Python 3.8+
- CUDA GPU (推荐V100 16GB+)
- vLLM 0.13.0+

### 安装依赖
```bash
pip install vllm pymupdf requests
```

### 使用示例
```python
from production_extractor_v7 import extract_pdf

result = extract_pdf("/path/to/medical.pdf")

if result["success"]:
    print(f"文档类型: {result['doc_type']}")
    print(f"耗时: {result['time']:.1f}s")
    print(f"提取结果: {result['result']}")
else:
    print(f"失败: {result['error']}")
```

## 输出格式

### GUIDELINE类型
```json
{
  "doc_metadata": {"title": "...", "organization": "...", "sources": ["p1"]},
  "scope": {"population": "...", "conditions": "...", "sources": ["px"]},
  "recommendations": [
    {"id": "1.1", "text": "推荐内容", "strength": "强度", "sources": ["px"]}
  ],
  "key_evidence": [{"id": "E1", "description": "...", "sources": ["px"]}]
}
```

### REVIEW类型
```json
{
  "doc_metadata": {"title": "...", "authors": "...", "journal": "...", "sources": ["p1"]},
  "scope": {"objective": "...", "sources": ["px"]},
  "key_findings": [{"id": "F1", "finding": "...", "sources": ["px"]}],
  "conclusions": [{"id": "C1", "text": "...", "sources": ["px"]}]
}
```

## 版本历史

| 版本 | 平均分 | 改进 |
|------|--------|------|
| v4 | 4.1 | 基线 |
| v6 | 7.1 | 静态few-shot |
| v7.2 | **9.6** | 动态few-shot + 健壮性优化 |

## License

MIT