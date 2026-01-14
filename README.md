# Medical PDF Structured Information Extractor

医学PDF结构化信息提取系统，基于Qwen3-8B模型，支持指南、综述、其他类型医学文献的自动化提取。

## 核心特性

- **智能文档分类**: 自动识别GUIDELINE/REVIEW/OTHER类型
- **动态Few-shot**: 根据文档类型加载对应高质量示例
- **分流架构**: 根据文档大小自动选择最优提取策略
- **结构化输出**: JSON格式，包含元数据、推荐/发现、证据、结论

## 当前版本

**v7.9** - 分流架构版

| 文档大小 | 策略 | 说明 |
|---------|------|------|
| ≤50页 | single_short | 智能选页，一次性提取 |
| 51-80页 | single_long | 选60页，一次性提取 |
| >80页 | mapreduce | 先选80页，分块提取后合并 |

## 性能指标

| 文档类型 | 平均评分 | 平均耗时 |
|---------|---------|---------|
| 短文档(≤15页) | 8.3/10 | 35s |
| 中文档(16-50页) | 6.5/10 | 45s |
| 长文档(>50页) | 6.0/10 | 90s |

## 快速开始

```python
from production_extractor_v79 import extract_pdf

result = extract_pdf("your_medical_paper.pdf")
print(result['doc_type'])  # GUIDELINE/REVIEW/OTHER
print(result['result'])    # 结构化提取结果
```

## 部署要求

- Python 3.10+
- vLLM (Qwen3-8B, fp16, max_model_len=16384)
- PyMuPDF (fitz)

## 文件结构

```
├── production_extractor_v79.py  # 主提取器（生产版）
├── fewshot_samples/             # Few-shot示例
│   ├── GUIDELINE_sample.json
│   ├── REVIEW_sample.json
│   └── OTHER_sample.json
├── test_v79.py                  # 测试脚本
└── docs/                        # 文档
```

## 待优化方向

### 1. Milvus语义检索集成（高优先级）
- 当前：启发式关键词选页
- 优化：用Milvus+BGE-M3/Qwen3-embedding语义检索选页
- 预期：长文档准确度提升1-2分

### 2. 提示词优化
- Quote-then-Structure：先摘录原文再结构化
- JSON Schema约束：更严格的输出格式
- 思维链引导

### 3. 长文档分块策略
- 先选后分块（已实现）
- 锚点窗口检索
- 自适应chunk大小

## 版本历史

| 版本 | 日期 | 评分 | 改进 |
|------|------|------|------|
| v7.2 | 01-08 | 9.6 | 动态Few-shot |
| v7.3 | 01-12 | 8.3 | 智能页面选择 |
| v7.6 | 01-12 | 7.5 | 混合策略 |
| v7.9 | 01-13 | 7.0 | 分流架构 |

## License

MIT
