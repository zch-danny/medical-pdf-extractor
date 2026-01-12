#!/bin/bash
# vLLM 优化版 - FP16 + prefix-caching（V100兼容）

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --served-model-name qwen3-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --dtype half \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --trust-remote-code
