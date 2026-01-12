#!/bin/bash
# DeepSeek-OCR vLLM 部署脚本 (Linux)
# 用于启动 DeepSeek-OCR 模型服务

# ==================== 配置 ====================
MODEL_NAME="${OCR_MODEL_NAME:-deepseek-ai/DeepSeek-OCR}"
HOST="${OCR_HOST:-0.0.0.0}"
PORT="${OCR_PORT:-8001}"  # 与摘要模型不同端口

# GPU 配置
GPU_MEMORY_UTIL="${OCR_GPU_UTIL:-0.4}"  # OCR 模型较小
MAX_MODEL_LEN="${OCR_MAX_LEN:-4096}"

echo "========================================"
echo "    DeepSeek-OCR vLLM 服务启动"
echo "========================================"
echo ""
echo "模型: $MODEL_NAME"
echo "地址: http://${HOST}:${PORT}"
echo "显存: $GPU_MEMORY_UTIL"
echo ""

# 启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype auto

# 量化版本启动示例:
# python -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_NAME" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --gpu-memory-utilization 0.3 \
#     --max-model-len 4096 \
#     --quantization awq \
#     --trust-remote-code
