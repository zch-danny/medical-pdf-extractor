#!/bin/bash
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=/root/autodl-tmp/tmp
export TRITON_CACHE_DIR=/root/autodl-tmp/triton_cache
export PROMETHEUS_MULTIPROC_DIR=/root/autodl-tmp/prometheus

# 完全禁用 Triton 相关功能
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ATTENTION_BACKEND=XFORMERS

rm -rf $PROMETHEUS_MULTIPROC_DIR/* $TRITON_CACHE_DIR/*

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --served-model-name qwen3-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --dtype half \
    --gpu-memory-utilization 0.92 \
    --enforce-eager \
    --trust-remote-code
