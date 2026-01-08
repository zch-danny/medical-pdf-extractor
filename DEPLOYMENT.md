# 医学PDF提取系统 - 部署指南

## 1. 环境要求

### 硬件要求
- GPU: NVIDIA V100 16GB 或更高
- 内存: 32GB+
- 存储: 50GB+ (模型缓存)

### 软件要求
- Ubuntu 20.04+
- Python 3.8+
- CUDA 11.8+
- cuDNN 8.6+

## 2. 安装步骤

### 2.1 安装Python依赖
```bash
pip install vllm==0.13.0
pip install pymupdf
pip install requests
```

### 2.2 下载模型
```bash
huggingface-cli download Qwen/Qwen3-8B
```

## 3. 启动服务

### 3.1 启动vLLM模型服务
```bash
vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --dtype float16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

### 3.2 验证服务状态
```bash
curl http://localhost:8000/v1/models
```

## 4. 使用方法

### 4.1 Python调用
```python
from production_extractor_v7 import extract_pdf

result = extract_pdf("/path/to/medical.pdf")

if result["success"]:
    print(f"类型: {result['doc_type']}")
    print(f"结果: {result['result']}")
else:
    print(f"错误: {result['error']}")
```

### 4.2 批量处理
```python
from production_extractor_v7 import MedicalPDFExtractor
from pathlib import Path

extractor = MedicalPDFExtractor()
pdf_dir = Path("/path/to/pdfs")

for pdf in pdf_dir.glob("*.pdf"):
    result = extractor.extract(str(pdf))
```

## 5. 配置说明

### 关键参数
**必须设置**: 调用Qwen3-8B时需要禁用thinking模式
```python
"chat_template_kwargs": {"enable_thinking": False}
```

## 6. 故障排除

**Q: API返回 "max_tokens is too large"**
A: 减少输入文本长度或降低max_tokens值

**Q: JSON解析失败**
A: v7.2已修复大部分问题，若仍出现，检查PDF是否损坏

**Q: 提取结果不完整**
A: 长文档会被截断，这是已知限制