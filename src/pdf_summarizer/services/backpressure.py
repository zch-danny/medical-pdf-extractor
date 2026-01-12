"""
背压控制服务

功能:
1. 限制并发请求数
2. 请求排队
3. 过载保护

参考 safety_llm_deploy 的背压控制机制
"""
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    from pdf_summarizer.core.high_performance import CONCURRENCY_CONFIG
except ImportError:
    CONCURRENCY_CONFIG = {
        "max_concurrent_requests": 20,
        "backpressure_threshold": 50,
        "max_queue_size": 100,
    }


@dataclass
class BackpressureStats:
    """背压统计"""
    current_requests: int = 0
    total_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    max_concurrent_seen: int = 0
    
    def to_dict(self) -> dict:
        return {
            "current_requests": self.current_requests,
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "queued_requests": self.queued_requests,
            "max_concurrent_seen": self.max_concurrent_seen,
            "rejection_rate": f"{self.rejected_requests / max(1, self.total_requests):.2%}",
        }


class BackpressureController:
    """
    背压控制器
    
    使用信号量限制并发，超过阈值拒绝请求
    """
    
    _instance: Optional["BackpressureController"] = None
    
    def __init__(
        self,
        max_concurrent: int = None,
        backpressure_threshold: int = None,
    ):
        self.max_concurrent = max_concurrent or CONCURRENCY_CONFIG["max_concurrent_requests"]
        self.backpressure_threshold = backpressure_threshold or CONCURRENCY_CONFIG["backpressure_threshold"]
        
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = BackpressureStats()
        self._lock = asyncio.Lock()
        
        logger.info(
            f"BackpressureController 初始化: "
            f"max_concurrent={self.max_concurrent}, "
            f"threshold={self.backpressure_threshold}"
        )
    
    @classmethod
    def get_instance(cls) -> "BackpressureController":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_semaphore(self) -> asyncio.Semaphore:
        """确保信号量已初始化"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        获取执行许可
        
        Args:
            timeout: 等待超时时间（秒）
            
        Returns:
            是否获取成功
        """
        self._stats.total_requests += 1
        
        # 检查是否超过背压阈值
        if self._stats.current_requests >= self.backpressure_threshold:
            self._stats.rejected_requests += 1
            logger.warning(
                f"背压触发，拒绝请求: current={self._stats.current_requests}, "
                f"threshold={self.backpressure_threshold}"
            )
            return False
        
        semaphore = self._ensure_semaphore()
        
        try:
            # 尝试在超时时间内获取信号量
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
            
            async with self._lock:
                self._stats.current_requests += 1
                self._stats.max_concurrent_seen = max(
                    self._stats.max_concurrent_seen,
                    self._stats.current_requests
                )
            
            return True
            
        except asyncio.TimeoutError:
            self._stats.rejected_requests += 1
            logger.warning(f"获取执行许可超时: timeout={timeout}s")
            return False
    
    async def release(self) -> None:
        """释放执行许可"""
        if self._semaphore:
            self._semaphore.release()
            
            async with self._lock:
                self._stats.current_requests = max(0, self._stats.current_requests - 1)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return self._stats.to_dict()
    
    def is_overloaded(self) -> bool:
        """是否过载"""
        return self._stats.current_requests >= self.backpressure_threshold


# 上下文管理器
class BackpressureContext:
    """背压控制上下文管理器"""
    
    def __init__(self, controller: BackpressureController, timeout: float = 30.0):
        self.controller = controller
        self.timeout = timeout
        self.acquired = False
    
    async def __aenter__(self):
        self.acquired = await self.controller.acquire(self.timeout)
        if not self.acquired:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="服务繁忙，请稍后重试",
                headers={"Retry-After": "30"}
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            await self.controller.release()
        return False
