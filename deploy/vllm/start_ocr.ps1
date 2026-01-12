# DeepSeek-OCR vLLM 部署脚本 (Windows PowerShell)
# 用于启动 DeepSeek-OCR 模型服务

# ==================== 配置 ====================
$MODEL_NAME = if ($env:OCR_MODEL_NAME) { $env:OCR_MODEL_NAME } else { "deepseek-ai/DeepSeek-OCR" }
$SERVER_HOST = if ($env:OCR_HOST) { $env:OCR_HOST } else { "0.0.0.0" }
$SERVER_PORT = if ($env:OCR_PORT) { $env:OCR_PORT } else { "8001" }  # 与摘要模型不同端口

# GPU 配置
$GPU_MEMORY_UTIL = if ($env:OCR_GPU_UTIL) { $env:OCR_GPU_UTIL } else { "0.4" }  # OCR 模型较小
$MAX_MODEL_LEN = if ($env:OCR_MAX_LEN) { $env:OCR_MAX_LEN } else { "4096" }

# 量化配置 (可选)
# 如果显存不足，可以使用量化版本
# $QUANTIZATION = "--quantization awq"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    DeepSeek-OCR vLLM 服务启动" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "模型: $MODEL_NAME" -ForegroundColor Yellow
Write-Host "地址: http://${SERVER_HOST}:${SERVER_PORT}" -ForegroundColor Yellow
Write-Host "显存: $GPU_MEMORY_UTIL" -ForegroundColor Yellow
Write-Host ""

# 启动 vLLM
python -m vllm.entrypoints.openai.api_server `
    --model $MODEL_NAME `
    --host $SERVER_HOST `
    --port $SERVER_PORT `
    --gpu-memory-utilization $GPU_MEMORY_UTIL `
    --max-model-len $MAX_MODEL_LEN `
    --trust-remote-code `
    --dtype auto

# 量化版本启动示例:
# python -m vllm.entrypoints.openai.api_server `
#     --model $MODEL_NAME `
#     --host $HOST `
#     --port $PORT `
#     --gpu-memory-utilization 0.3 `
#     --max-model-len 4096 `
#     --quantization awq `
#     --trust-remote-code
