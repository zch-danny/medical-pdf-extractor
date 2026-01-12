"""
依赖注入容器

集中管理所有服务依赖，支持测试时依赖替换
"""

from typing import Optional, Dict, Any, Type, TypeVar, Callable, AsyncGenerator
from dataclasses import dataclass, field
import threading
from loguru import logger

T = TypeVar("T")


@dataclass
class ServiceRegistry:
    """
    服务注册表
    
    集中管理所有服务实例，支持:
    - 懒加载
    - 依赖替换（用于测试）
    - 生命周期管理
    """
    
    _services: Dict[str, Any] = field(default_factory=dict)
    _factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)
    _overrides: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def register(self, name: str, factory: Callable[[], T]) -> None:
        """
        注册服务工厂
        
        Args:
            name: 服务名称
            factory: 服务工厂函数
        """
        with self._lock:
            self._factories[name] = factory
            logger.debug(f"服务已注册: {name}")
    
    def get(self, name: str) -> Any:
        """
        获取服务实例
        
        Args:
            name: 服务名称
            
        Returns:
            服务实例
        """
        # 优先返回覆盖的服务
        if name in self._overrides:
            return self._overrides[name]
        
        # 懒加载服务
        if name not in self._services:
            with self._lock:
                if name not in self._services:
                    if name not in self._factories:
                        raise KeyError(f"服务未注册: {name}")
                    self._services[name] = self._factories[name]()
                    logger.debug(f"服务已初始化: {name}")
        
        return self._services[name]
    
    def override(self, name: str, instance: Any) -> None:
        """
        覆盖服务实例（用于测试）
        
        Args:
            name: 服务名称
            instance: 替换的实例
        """
        with self._lock:
            self._overrides[name] = instance
            logger.debug(f"服务已覆盖: {name}")
    
    def clear_override(self, name: str) -> None:
        """清除服务覆盖"""
        with self._lock:
            self._overrides.pop(name, None)
    
    def clear_all_overrides(self) -> None:
        """清除所有服务覆盖"""
        with self._lock:
            self._overrides.clear()
    
    def reset(self) -> None:
        """重置所有服务实例（用于测试）"""
        with self._lock:
            self._services.clear()
            self._overrides.clear()


# 全局服务注册表
_registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """获取全局服务注册表"""
    return _registry


# ===========================================
# FastAPI 依赖函数
# ===========================================

def get_llm_client():
    """获取 LLM 客户端依赖"""
    from pdf_summarizer.models.llm_client import LLMClient
    
    if "llm_client" not in _registry._factories:
        _registry.register("llm_client", LLMClient)
    
    return _registry.get("llm_client")


def get_summarizer_service():
    """获取摘要服务依赖"""
    from pdf_summarizer.services.summarizer import SummarizerService
    
    if "summarizer_service" not in _registry._factories:
        _registry.register("summarizer_service", SummarizerService)
    
    return _registry.get("summarizer_service")


def get_task_manager():
    """获取任务管理器依赖"""
    from pdf_summarizer.services.task_manager import TaskManager
    
    if "task_manager" not in _registry._factories:
        _registry.register("task_manager", TaskManager.get_instance)
    
    return _registry.get("task_manager")


def get_cache():
    """获取缓存服务依赖"""
    from pdf_summarizer.utils.tiered_cache import TieredCache
    
    if "cache" not in _registry._factories:
        _registry.register("cache", TieredCache.get_instance)
    
    return _registry.get("cache")


def get_rate_limiter():
    """获取限流器依赖"""
    from pdf_summarizer.api.rate_limit import RateLimiter
    
    if "rate_limiter" not in _registry._factories:
        _registry.register("rate_limiter", RateLimiter.get_instance)
    
    return _registry.get("rate_limiter")


def get_backpressure_controller():
    """获取背压控制器依赖"""
    from pdf_summarizer.services.backpressure import BackpressureController
    
    if "backpressure" not in _registry._factories:
        _registry.register("backpressure", BackpressureController.get_instance)
    
    return _registry.get("backpressure")


def get_alert_service():
    """获取告警服务依赖"""
    from pdf_summarizer.services.alert_service import AlertService
    
    if "alert_service" not in _registry._factories:
        _registry.register("alert_service", AlertService.get_instance)
    
    return _registry.get("alert_service")


# ===========================================
# 依赖注入上下文管理器（用于测试）
# ===========================================

class DependencyOverride:
    """
    依赖覆盖上下文管理器
    
    使用方法:
        with DependencyOverride("llm_client", mock_client):
            # 测试代码
            pass
    """
    
    def __init__(self, name: str, instance: Any):
        self.name = name
        self.instance = instance
    
    def __enter__(self):
        _registry.override(self.name, self.instance)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _registry.clear_override(self.name)
        return False


def override_dependency(name: str, instance: Any) -> DependencyOverride:
    """
    依赖覆盖辅助函数
    
    Args:
        name: 依赖名称
        instance: 替换实例
        
    Returns:
        上下文管理器
    """
    return DependencyOverride(name, instance)
