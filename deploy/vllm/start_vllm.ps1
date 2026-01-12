# =================================================================
# vLLM 启动脚本 (Windows PowerShell) - Qwen3-8B
# =================================================================

# 配置参数
$MODEL_NAME = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "Qwen/Qwen3-8B" }
$HOST = if ($env:VLLM_HOST) { $env:VLLM_HOST } else { "0.0.0.0" }
$PORT = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$MAX_MODEL_LEN = if ($env:MAX_MODEL_LEN) { $env:MAX_MODEL_LEN } else { "32768" }
$GPU_MEMORY_UTILIZATION = if ($env:GPU_MEMORY_UTILIZATION) { $env:GPU_MEMORY_UTILIZATION } else { "0.9" }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "vLLM Server Configuration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Model: $MODEL_NAME"
Write-Host "Host: ${HOST}:${PORT}"
Write-Host "Max Model Length: $MAX_MODEL_LEN"
Write-Host "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
Write-Host "==========================================" -ForegroundColor Cyan

# 启动 vLLM
python -m vllm.entrypoints.openai.api_server `
    --model $MODEL_NAME `
    --host $HOST `
    --port $PORT `
    --max-model-len $MAX_MODEL_LEN `
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION `
    --trust-remote-code `
    --enable-prefix-caching
