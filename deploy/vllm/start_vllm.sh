#!/bin/bash
# =================================================================
# vLLM 启动脚本 - Qwen3-8B
# =================================================================

# 配置
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "=========================================="
echo "vLLM Server Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Host: $HOST:$PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "=========================================="

# 启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests
