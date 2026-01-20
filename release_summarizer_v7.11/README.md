# Medical PDF Processing System

医学PDF处理系统，基于Qwen3-8B模型，支持医学文献的结构化信息提取和智能摘要生成。

## 核心模块

### 1. 结构化信息提取器 (Extractor)
- **智能文档分类**: 自动识别GUIDELINE/REVIEW/OTHER类型
- **动态Few-shot**: 根据文档类型加载对应高质量示例
- **分流架构**: 根据文档大小自动选择最优提取策略
- **结构化输出**: JSON格式，包含元数据、推荐/发现、证据、结论

### 2. 智能摘要生成器 (Summarizer) ⭐ 推荐
- **智能内容提取**: 自动跳过目录，提取正文核心内容
- **严格约束**: 禁止臆测、扩写，只总结原文明确内容
- **高准确度**: GPT-5.2评估平均7.5+分
- **稳定输出**: temperature=0.05确保一致性

## 当前版本

| 模块 | 版本 | 评分 | 说明 |
|------|------|------|------|
| 摘要生成器 | v7.11 | 7.5+ | 智能内容提取+严格约束 |
| 信息提取器 | v7.9 | 7.0 | 分流架构 |

## 摘要生成器快速开始

```python
from medical_summarizer_v7_11 import MedicalSummarizer

summarizer = MedicalSummarizer()
result = summarizer.generate_summary("your_medical_paper.pdf")

print(result['summary'])      # 摘要内容
print(result['key_points'])   # 关键要点
print(result['keywords'])     # 提取的关键词
```

### 摘要生成器特性
- **智能内容提取**: 对长文档自动跳过目录，定位正文
- **严格禁止臆测**: 只总结原文明确写出的内容
- **后处理过滤**: 自动移除可能的臆测日期等
- **温度控制**: temperature=0.05确保输出稳定

## 信息提取器快速开始

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
- jieba (用于关键词提取)

## 文件结构

```
├── medical_summarizer_v7_11.py  # 摘要生成器（推荐）
├── production_extractor_v79.py  # 信息提取器
├── evaluate_summarizer.py       # 摘要质量评估工具
├── fewshot_samples/             # Few-shot示例
└── docs/                        # 文档
```

## 版本历史

### 摘要生成器
| 版本 | 日期 | 评分 | 改进 |
|------|------|------|------|
| v7.11 | 01-20 | 7.5+ | 智能内容提取+严格约束+temperature=0.05 |
| v7.8 | 01-19 | 7.3 | 简洁约束+后处理 |
| v7.6 | 01-19 | 6.9 | 单次调用策略 |

### 信息提取器
| 版本 | 日期 | 评分 | 改进 |
|------|------|------|------|
| v7.9 | 01-13 | 7.0 | 分流架构 |
| v7.6 | 01-12 | 7.5 | 混合策略 |
| v7.3 | 01-12 | 8.3 | 智能页面选择 |

## License

MIT
