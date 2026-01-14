# 生产环境部署指南

## 环境要求
- GPU: NVIDIA V100 16GB+
- 内存: 32GB+
- Python: 3.10+
- CUDA: 11.8+

## 安装步骤
```bash
pip install vllm==0.13.0 pymupdf requests
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

## 启动服务
```bash
vllm serve ./models/Qwen3-8B \
    --port 8000 --dtype float16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --chat-template-kwargs '{"enable_thinking": false}'
```

## 验证
```bash
curl http://localhost:8000/v1/models
python production_extractor_v79.py test.pdf
```

## Systemd服务
```bash
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Service
[Service]
ExecStart=/path/to/vllm serve ...
Restart=on-failure
[Install]
WantedBy=multi-user.target
```
