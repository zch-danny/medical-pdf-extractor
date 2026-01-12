# Medical PDF Extractor v7.6

医学PDF结构化信息提取系统，使用Qwen3-8B模型进行文档分类和信息提取。

## 特性

- **智能文档分类**: 自动识别GUIDELINE/REVIEW/OTHER类型
- **混合页面策略**: 根据文档大小自适应选择最优提取策略
- **动态Few-shot**: 根据文档类型加载对应高质量示例
- **高准确率**: GPT-5.2评分9.0+，≥8分占比100%

## 混合策略

| 文档大小 | 页面策略 |
|---------|---------|
| ≤15页 (短文档) | 全部保留，只跳过空白页 |
| 16-50页 (中等) | 跳过目录/参考文献/空白页 |
| >50页 (长文档) | 智能选择最多50页关键内容 |

## 快速开始

### 1. 启动vLLM服务
```bash
bash /root/autodl-tmp/vllm_server_v13.sh
# 等待约2分钟模型加载
curl http://localhost:8000/v1/models
```

### 2. 提取PDF
```python
from production_extractor_v7 import extract_pdf

result = extract_pdf("/path/to/medical.pdf")
if result["success"]:
    print(f"类型: {result['doc_type']}")
    print(f"文档大小: {result['stats']['doc_size']}")
    print(f"结果: {result['result']}")
```

## 输出格式

```json
{
  "success": true,
  "doc_type": "GUIDELINE",
  "result": {
    "doc_metadata": {"title": "...", "authors": "...", "sources": ["p1"]},
    "scope": {"content_summary": "...", "sources": ["p2"]},
    "recommendations": [
      {"id": "1.1", "text": "推荐内容", "strength": "强", "sources": ["p5"]}
    ],
    "key_evidence": [...]
  },
  "stats": {
    "total_pages": 27,
    "selected_pages": 27,
    "doc_size": "medium"
  },
  "time": 42.5
}
```

## 文件结构

```
production_extractor_v7.py   # 主提取器(v7.6混合策略)
fewshot_samples/
  ├── GUIDELINE_sample.json
  ├── REVIEW_sample.json
  └── OTHER_sample.json
```

## 性能指标

| 指标 | 结果 |
|------|------|
| 成功率 | 100% |
| GPT-5.2评分 | 9.0+ |
| ≥8分占比 | 100% |
| 平均耗时 | ~45秒 |

## 版本历史

- **v7.6** (2026-01-12): 混合策略，兼顾短文档高评分和长文档支持
- **v7.3** (2026-01-12): 智能页面选择，支持长文档
- **v7.2** (2026-01-08): 动态Few-shot，评分9.60

## 依赖

- Python 3.8+
- vLLM 0.13.0
- PyMuPDF (fitz)
- requests
