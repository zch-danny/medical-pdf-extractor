"""
熔断器模块

实现熔断器模式 (Circuit Breaker Pattern):
- closed: 正常状态，请求正常处理
- open: 熔断状态，直接拒绝请求
- half-open: 半开状态，尝试恢复

功能:
- 失败率阈值触发熔断
- 自动恢复检测
- 熔断状态通知
"""

import asyncio
import time
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from loguru import logger


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 触发熔断的失败次数
    success_threshold: int = 3          # 恢复所需的成功次数
    timeout_seconds: float = 30.0       # 熔断超时时间（秒）
    window_size: int = 10               # 滑动窗口大小
    failure_rate_threshold: float = 0.5  # 失败率阈值


@dataclass 
class CircuitBreakerStats:
    """熔断器统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "state_changes": self.state_changes,
            "success_rate": f"{self.successful_requests / max(1, self.total_requests):.2%}",
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreaker:
    """
    熔断器
    
    使用方法:
        circuit_breaker = CircuitBreaker("llm_service")
        
        @circuit_breaker
        async def call_llm(...):
            ...
    
    或者:
        async with circuit_breaker:
            result = await call_llm(...)
    """
    
    _instances: Dict[str, "CircuitBreaker"] = {}
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
        
        # 滑动窗口记录
        self._results: deque = deque(maxlen=self.config.window_size)
        
        # 统计
        self._stats = CircuitBreakerStats()
        
        # 锁
        self._lock = asyncio.Lock()
        
        logger.info(f"CircuitBreaker[{name}] 初始化: threshold={self.config.failure_threshold}")
    
    @classmethod
    def get_instance(cls, name: str, config: Optional[CircuitBreakerConfig] = None) -> "CircuitBreaker":
        """获取命名的熔断器实例"""
        if name not in cls._instances:
            cls._instances[name] = cls(name, config)
        return cls._instances[name]
    
    @property
    def state(self) -> CircuitState:
        """获取当前状态"""
        return self._state
    
    @property
    def is_open(self) -> bool:
        """是否处于熔断状态"""
        return self._state == CircuitState.OPEN
    
    @property
    def failure_rate(self) -> float:
        """当前失败率"""
        if not self._results:
            return 0.0
        failures = sum(1 for r in self._results if not r)
        return failures / len(self._results)
    
    async def _should_allow_request(self) -> bool:
        """判断是否应该允许请求"""
        async with self._lock:
            now = time.time()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # 检查是否超过熔断超时时间
                if now - self._last_state_change >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(f"CircuitBreaker[{self.name}] 进入半开状态")
                    return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                # 半开状态只允许有限的请求通过
                return True
            
            return False
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._last_state_change = time.time()
            self._stats.state_changes += 1
            
            if new_state == CircuitState.HALF_OPEN:
                self._success_count = 0
            
            logger.warning(
                f"CircuitBreaker[{self.name}] 状态变更: {old_state.value} -> {new_state.value}"
            )
    
    async def record_success(self) -> None:
        """记录成功"""
        async with self._lock:
            self._results.append(True)
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    logger.info(f"CircuitBreaker[{self.name}] 恢复正常")
    
    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """记录失败"""
        async with self._lock:
            self._results.append(False)
            self._failure_count += 1
            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下失败立即熔断
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"CircuitBreaker[{self.name}] 半开状态失败，重新熔断")
            
            elif self._state == CircuitState.CLOSED:
                # 检查是否应该熔断
                if (self._failure_count >= self.config.failure_threshold or
                    self.failure_rate >= self.config.failure_rate_threshold):
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"CircuitBreaker[{self.name}] 触发熔断: "
                        f"failures={self._failure_count}, rate={self.failure_rate:.2%}"
                    )
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        allowed = await self._should_allow_request()
        if not allowed:
            self._stats.rejected_requests += 1
            raise CircuitBreakerOpenError(
                f"CircuitBreaker[{self.name}] 处于熔断状态"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_val)
        return False
    
    def __call__(self, func: Callable) -> Callable:
        """装饰器模式"""
        async def wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return wrapper
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_rate": f"{self.failure_rate:.2%}",
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "stats": self._stats.to_dict(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            }
        }
    
    def reset(self) -> None:
        """重置熔断器（用于测试）"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._results.clear()
        self._stats = CircuitBreakerStats()


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


# ===========================================
# 预置熔断器实例
# ===========================================

def get_llm_circuit_breaker() -> CircuitBreaker:
    """获取 LLM 服务熔断器"""
    return CircuitBreaker.get_instance(
        "llm_service",
        CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=60.0,
            window_size=20,
            failure_rate_threshold=0.5,
        )
    )


def get_ocr_circuit_breaker() -> CircuitBreaker:
    """获取 OCR 服务熔断器"""
    return CircuitBreaker.get_instance(
        "ocr_service",
        CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            window_size=10,
            failure_rate_threshold=0.5,
        )
    )
