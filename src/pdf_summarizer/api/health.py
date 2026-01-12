"""
健康检查端点模块

提供 Kubernetes 风格的健康检查端点：
- /health/live - 存活探针
- /health/ready - 就绪探针
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    """健康状态响应模型"""
    status: str
    timestamp: str
    version: str = "1.0.0"
    details: Optional[Dict[str, Any]] = None


class ReadinessStatus(BaseModel):
    """就绪状态响应模型"""
    status: str
    timestamp: str
    checks: Dict[str, Dict[str, Any]]


# 服务状态追踪
class ServiceHealthTracker:
    """服务健康状态追踪器"""
    
    def __init__(self):
        self._is_ready = False
        self._startup_time: Optional[datetime] = None
        self._last_check_time: Optional[datetime] = None
        self._vllm_healthy = False
        self._redis_healthy = False
    
    def mark_ready(self):
        """标记服务就绪"""
        self._is_ready = True
        self._startup_time = datetime.now()
    
    def mark_not_ready(self):
        """标记服务未就绪"""
        self._is_ready = False
    
    @property
    def is_ready(self) -> bool:
        return self._is_ready
    
    @property
    def startup_time(self) -> Optional[datetime]:
        return self._startup_time
    
    def update_vllm_status(self, healthy: bool):
        """更新vLLM服务状态"""
        self._vllm_healthy = healthy
    
    def update_redis_status(self, healthy: bool):
        """更新Redis状态"""
        self._redis_healthy = healthy
    
    @property
    def vllm_healthy(self) -> bool:
        return self._vllm_healthy
    
    @property
    def redis_healthy(self) -> bool:
        return self._redis_healthy


# 全局健康追踪器
health_tracker = ServiceHealthTracker()


@router.get("/health")
async def health_check() -> HealthStatus:
    """
    基础健康检查
    
    返回服务基本运行状态
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        details={
            "uptime_seconds": (
                (datetime.now() - health_tracker.startup_time).total_seconds()
                if health_tracker.startup_time else 0
            ),
        }
    )


@router.get("/health/live")
async def liveness_probe(response: Response) -> Dict[str, Any]:
    """
    存活探针 (Liveness Probe)
    
    用于 Kubernetes 判断容器是否存活
    只要进程在运行，就返回 200
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health/ready")
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """
    就绪探针 (Readiness Probe)
    
    用于 Kubernetes 判断服务是否可以接收流量
    检查所有依赖服务的连接状态
    """
    checks = {}
    all_healthy = True
    
    # 检查 vLLM 服务
    vllm_status = await check_vllm_health()
    checks["vllm"] = vllm_status
    if not vllm_status["healthy"]:
        all_healthy = False
    
    # 检查 Redis（如果配置了）
    redis_status = await check_redis_health()
    checks["redis"] = redis_status
    # Redis 是可选的，不影响整体就绪状态
    
    # 设置响应状态码
    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
    }


async def check_vllm_health() -> Dict[str, Any]:
    """
    检查 vLLM 服务健康状态
    """
    try:
        import httpx
        from pdf_summarizer.core.config import settings
        
        vllm_url = getattr(settings, 'VLLM_API_URL', 'http://localhost:8000')
        health_url = f"{vllm_url}/health"
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(health_url)
            healthy = resp.status_code == 200
            health_tracker.update_vllm_status(healthy)
            return {
                "healthy": healthy,
                "response_time_ms": resp.elapsed.total_seconds() * 1000 if hasattr(resp, 'elapsed') else None,
                "status_code": resp.status_code,
            }
    except Exception as e:
        health_tracker.update_vllm_status(False)
        return {
            "healthy": False,
            "error": str(e),
        }


async def check_redis_health() -> Dict[str, Any]:
    """
    检查 Redis 连接健康状态
    """
    try:
        from pdf_summarizer.core.config import settings
        
        redis_url = getattr(settings, 'REDIS_URL', None)
        if not redis_url:
            return {
                "healthy": True,
                "status": "not_configured",
                "message": "Redis is optional and not configured",
            }
        
        import redis.asyncio as aioredis
        
        client = aioredis.from_url(redis_url, socket_timeout=2.0)
        await client.ping()
        await client.close()
        
        health_tracker.update_redis_status(True)
        return {
            "healthy": True,
            "status": "connected",
        }
    except ImportError:
        return {
            "healthy": True,
            "status": "not_available",
            "message": "Redis client not installed",
        }
    except Exception as e:
        health_tracker.update_redis_status(False)
        return {
            "healthy": False,
            "error": str(e),
        }


def get_health_tracker() -> ServiceHealthTracker:
    """获取健康追踪器实例"""
    return health_tracker
