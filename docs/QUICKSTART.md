# 快速开始指南

## 前置条件
- Python 3.10+
- NVIDIA GPU (16GB+) 或使用硅基流动云端API

## 步骤1: 安装依赖
```bash
pip install vllm pymupdf requests
```

## 步骤2: 启动vLLM服务
```bash
vllm serve Qwen/Qwen3-8B --port 8000 --dtype float16 --max-model-len 16384
```

## 步骤3: 提取PDF
```python
from production_extractor_v79 import extract_pdf
result = extract_pdf("document.pdf")
print(result)
```

## 使用云端API (无需GPU)
```bash
export SILICONFLOW_API_KEY=sk-xxx
python siliconflow_client.py document.pdf
```
