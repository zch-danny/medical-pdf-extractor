#!/bin/bash
# vLLM 优化版启动脚本 - AWQ量化 + 性能优化

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/hf_cache/Qwen3-8B-AWQ \
    --served-model-name qwen3-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --dtype half \
    --quantization awq \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --trust-remote-code
