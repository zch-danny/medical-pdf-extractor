#!/bin/bash
# vLLM 简化版启动脚本

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --served-model-name qwen3-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code
