"""
高性能配置 - PDF摘要服务优化参数

集中管理所有性能相关配置，便于调优和监控
参考 safety_llm_deploy 的高性能架构
"""
import os
from typing import Optional


# ==================== vLLM 服务端配置 ====================
VLLM_CONFIG = {
    # 基础配置
    "api_base": os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
    "model_name": os.getenv("MODEL_NAME", "Qwen/Qwen3-8B"),
    
    # 高吞吐参数
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "32")),  # 最大并发序列
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "32768")),  # 上下文长度
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_UTIL", "0.90")),
    
    # 推理参数
    "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "2048")),  # 摘要生成token上限
    "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.3")),  # 摘要适中温度
}


# ==================== API 客户端配置 ====================
API_CLIENT_CONFIG = {
    # HTTP 连接池配置
    "max_connections": int(os.getenv("API_MAX_CONNECTIONS", "50")),
    "max_keepalive": int(os.getenv("API_MAX_KEEPALIVE", "20")),
    "keepalive_expiry": float(os.getenv("API_KEEPALIVE_EXPIRY", "30.0")),
    
    # 超时配置 (秒) - 摘要生成较慢，需要更长超时
    "connect_timeout": float(os.getenv("API_CONNECT_TIMEOUT", "10.0")),
    "read_timeout": float(os.getenv("API_READ_TIMEOUT", "120.0")),  # 摘要生成可能较慢
    "write_timeout": float(os.getenv("API_WRITE_TIMEOUT", "10.0")),
    "pool_timeout": float(os.getenv("API_POOL_TIMEOUT", "10.0")),
    
    # 重试配置
    "max_retries": int(os.getenv("API_MAX_RETRIES", "3")),
    "retry_delay": float(os.getenv("API_RETRY_DELAY", "0.5")),
}


# ==================== 并发控制配置 ====================
CONCURRENCY_CONFIG = {
    # 最大并发请求数
    "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "20")),
    
    # 背压阈值 - 超过此值开始拒绝请求
    "backpressure_threshold": int(os.getenv("BACKPRESSURE_THRESHOLD", "50")),
    
    # 请求队列大小
    "max_queue_size": int(os.getenv("MAX_QUEUE_SIZE", "100")),
}


# ==================== 分层缓存配置 ====================
TIERED_CACHE_CONFIG = {
    # L1: 进程内 LRU 缓存 (最快, <0.1ms)
    "l1": {
        "enabled": os.getenv("CACHE_L1_ENABLED", "true").lower() == "true",
        "max_size": int(os.getenv("CACHE_L1_SIZE", "500")),  # PDF摘要较大，缓存条目少一些
        "ttl_seconds": int(os.getenv("CACHE_L1_TTL", "600")),  # 10分钟
    },
    
    # L2: Redis 缓存 (较快, <5ms)
    "l2": {
        "enabled": os.getenv("CACHE_L2_ENABLED", "true").lower() == "true",
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "password": os.getenv("REDIS_PASSWORD", "") or None,
        "key_prefix": os.getenv("REDIS_KEY_PREFIX", "pdf_sum:"),
        "max_size": int(os.getenv("CACHE_L2_SIZE", "10000")),
        "ttl_seconds": int(os.getenv("CACHE_L2_TTL", "3600")),  # 1小时
    },
}


# ==================== 限流配置 ====================
RATE_LIMIT_CONFIG = {
    "enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
    "requests_per_minute": int(os.getenv("RATE_LIMIT_RPM", "30")),  # PDF处理较重
    "requests_per_hour": int(os.getenv("RATE_LIMIT_RPH", "300")),
    "burst_size": int(os.getenv("RATE_LIMIT_BURST", "5")),
}


# ==================== 告警配置 ====================
ALERT_CONFIG = {
    "enabled": os.getenv("ALERT_ENABLED", "false").lower() == "true",
    "webhook_url": os.getenv("ALERT_WEBHOOK_URL", ""),
    
    # 告警阈值
    "error_rate_threshold": float(os.getenv("ALERT_ERROR_RATE", "0.1")),  # 10%错误率
    "latency_threshold_ms": int(os.getenv("ALERT_LATENCY_MS", "30000")),  # 30秒
    "queue_size_threshold": int(os.getenv("ALERT_QUEUE_SIZE", "50")),
}


# ==================== PDF处理配置 ====================
PDF_CONFIG = {
    "max_size_mb": int(os.getenv("MAX_PDF_SIZE_MB", "50")),
    "max_pages": int(os.getenv("MAX_PDF_PAGES", "200")),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "8000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "500")),
}
