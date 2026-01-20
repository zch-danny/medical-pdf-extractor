# Medical PDF Processing System

医学PDF处理系统，基于Qwen3-8B模型，支持医学文献的智能摘要生成。

## 核心模块

### 智能摘要生成器 (Summarizer)
- **智能内容提取**: 自动跳过目录，提取正文核心内容
- **严格约束**: 禁止臆测、扩写，只总结原文明确内容
- **高稳定性**: seed固定 + temperature=0.05确保一致性
- **高准确度**: GPT-5.2评估平均7.37分，标准差仅0.06

## 当前版本

| 模块 | 版本 | 评分 | 稳定性 |
|------|------|------|--------|
| 摘要生成器 | v7.16 | 7.37±0.06 | ★★★★★ |

## 快速开始

```python
from medical_summarizer_v7_16 import MedicalSummarizer

summarizer = MedicalSummarizer()
result = summarizer.generate_summary("your_medical_paper.pdf")

print(result['summary'])      # 摘要内容
print(result['key_points'])   # 关键要点
print(result['keywords'])     # 提取的关键词
```

## 摘要生成器特性

- **智能内容提取**: 对长文档自动跳过目录(>5000字符)，定位正文
- **严格禁止臆测**: 只总结原文明确写出的内容
- **后处理过滤**: 自动移除可能的臆测日期
- **确定性输出**: seed=42 + temperature=0.05

## 性能指标

基于3个测试PDF的多次评估（GPT-5.2评分）：

| 指标 | 得分 |
|------|------|
| 准确度 | 8.0 |
| 完整性 | 5.7 |
| 可读性 | 8.3 |
| **综合** | **7.37** |

## 部署要求

- Python 3.10+
- vLLM (Qwen3-8B, fp16, max_model_len=16384)
- PyMuPDF (fitz)
- jieba (用于关键词提取)

## 文件结构

```
├── medical_summarizer_v7_16.py  # 摘要生成器（生产版本）
├── evaluate_summarizer.py       # 摘要质量评估工具
└── README.md
```

## 版本历史

| 版本 | 日期 | 评分 | 稳定性 | 改进 |
|------|------|------|--------|------|
| v7.16 | 01-20 | 7.37 | 0.06 | seed固定+参数对齐（生产版本）|
| v7.11 | 01-20 | 7.24 | 0.15 | 智能内容提取+严格约束 |
| v7.8 | 01-19 | 7.3 | - | 简洁约束+后处理 |

## License

MIT
