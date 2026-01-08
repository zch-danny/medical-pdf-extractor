# Medical PDF Extractor

基于 Qwen3-8B 的医学文献结构化信息提取系统。

## 功能
- 自动分类医学文献类型（GUIDELINE/REVIEW/META/OTHER等）
- 根据类型选择专项提取器
- 提取结构化信息（元数据、推荐意见、关键发现等）

## 测试结果
| 版本 | 平均评分 | ≥8分占比 |
|------|---------|----------|
| v4 | 4.1/10 | 0% |
| v5 | 6.5/10 | 0% |
| **v6** | **7.1/10** | **20%** |

## 目录结构
```
├── prompts/
│   ├── classifier/          # 分类器提示词
│   └── extractors/          # 专项提取器
│       ├── GUIDELINE_extractor_v6.md
│       ├── OTHER_extractor_v4.md
│       └── REVIEW_extractor_v2.md
├── scripts/
│   └── batch_test.py        # 批量测试脚本
└── VERSION_LOG.md           # 版本日志
```

## 使用方法

### 1. 启动 vLLM 服务
```bash
nohup vllm serve Qwen/Qwen3-8B --port 8000 &
```

### 2. 运行测试
```bash
python scripts/batch_test.py
```

## 依赖
- Python 3.10+
- vLLM 0.13+
- PyMuPDF (fitz)
- requests

## License
MIT