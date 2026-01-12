"""
请求限流模块

基于滑动窗口的限流实现
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import Request, HTTPException
from loguru import logger


@dataclass
class RateLimitConfig:
    """限流配置"""
    requests_per_minute: int = 60      # 每分钟请求数
    requests_per_hour: int = 500       # 每小时请求数
    burst_size: int = 10               # 突发请求数
    enabled: bool = True


@dataclass
class ClientState:
    """客户端状态"""
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)
    last_request: float = 0


class RateLimiter:
    """
    滑动窗口限流器
    
    支持：
    - 每分钟限制
    - 每小时限制
    - 突发限制
    """
    
    _instance: Optional["RateLimiter"] = None
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._clients: Dict[str, ClientState] = defaultdict(ClientState)
        self._cleanup_interval = 300  # 5分钟清理一次
        self._last_cleanup = time.time()
    
    @classmethod
    def get_instance(cls, config: Optional[RateLimitConfig] = None) -> "RateLimiter":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用 X-Forwarded-For
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # 否则使用客户端 IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _cleanup_old_requests(self, state: ClientState, now: float):
        """清理过期请求记录"""
        minute_ago = now - 60
        hour_ago = now - 3600
        
        state.minute_requests = [
            t for t in state.minute_requests if t > minute_ago
        ]
        state.hour_requests = [
            t for t in state.hour_requests if t > hour_ago
        ]
    
    def _maybe_cleanup_clients(self):
        """定期清理不活跃的客户端"""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            inactive_threshold = now - 3600  # 1小时不活跃
            inactive_clients = [
                cid for cid, state in self._clients.items()
                if state.last_request < inactive_threshold
            ]
            for cid in inactive_clients:
                del self._clients[cid]
            
            if inactive_clients:
                logger.debug(f"清理 {len(inactive_clients)} 个不活跃客户端")
            
            self._last_cleanup = now
    
    def check_rate_limit(self, request: Request) -> Tuple[bool, Optional[str]]:
        """
        检查请求是否超过限流
        
        Returns:
            (是否允许, 错误信息)
        """
        if not self.config.enabled:
            return True, None
        
        now = time.time()
        client_id = self._get_client_id(request)
        state = self._clients[client_id]
        
        # 清理过期记录
        self._cleanup_old_requests(state, now)
        self._maybe_cleanup_clients()
        
        # 检查每分钟限制
        if len(state.minute_requests) >= self.config.requests_per_minute:
            wait_time = 60 - (now - state.minute_requests[0])
            return False, f"请求过于频繁，请 {int(wait_time)} 秒后重试"
        
        # 检查每小时限制
        if len(state.hour_requests) >= self.config.requests_per_hour:
            wait_time = 3600 - (now - state.hour_requests[0])
            return False, f"已达到小时限制，请 {int(wait_time/60)} 分钟后重试"
        
        # 检查突发限制（最近1秒内的请求）
        recent = [t for t in state.minute_requests if now - t < 1]
        if len(recent) >= self.config.burst_size:
            return False, "请求过于频繁，请稍后重试"
        
        # 记录请求
        state.minute_requests.append(now)
        state.hour_requests.append(now)
        state.last_request = now
        
        return True, None
    
    def get_remaining(self, request: Request) -> dict:
        """获取剩余配额"""
        client_id = self._get_client_id(request)
        state = self._clients[client_id]
        now = time.time()
        
        self._cleanup_old_requests(state, now)
        
        return {
            "minute_remaining": max(0, self.config.requests_per_minute - len(state.minute_requests)),
            "hour_remaining": max(0, self.config.requests_per_hour - len(state.hour_requests)),
            "minute_limit": self.config.requests_per_minute,
            "hour_limit": self.config.requests_per_hour,
        }


# 限流依赖
async def check_rate_limit(request: Request):
    """FastAPI 依赖：检查限流"""
    limiter = RateLimiter.get_instance()
    allowed, error_msg = limiter.check_rate_limit(request)
    
    if not allowed:
        logger.warning(f"限流触发: {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=429,
            detail=error_msg,
            headers={"Retry-After": "60"}
        )
    
    return True
