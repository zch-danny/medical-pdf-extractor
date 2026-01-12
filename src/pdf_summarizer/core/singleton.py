"""
统一单例模式实现

提供线程安全的单例装饰器，用于统一项目中所有服务的单例实现
"""

import threading
from typing import TypeVar, Type, Dict, Any, Optional, Callable
from functools import wraps

T = TypeVar("T")


class SingletonMeta(type):
    """
    线程安全的单例元类
    
    使用方法:
        class MyService(metaclass=SingletonMeta):
            pass
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # 双重检查锁定
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def clear_instance(mcs, cls: type) -> None:
        """清除指定类的单例实例（用于测试）"""
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]
    
    @classmethod
    def clear_all(mcs) -> None:
        """清除所有单例实例（用于测试）"""
        with mcs._lock:
            mcs._instances.clear()


def singleton(cls: Type[T]) -> Type[T]:
    """
    单例装饰器
    
    使用方法:
        @singleton
        class MyService:
            pass
    
    Args:
        cls: 要装饰的类
        
    Returns:
        单例包装后的类
    """
    instances: Dict[type, T] = {}
    lock = threading.Lock()
    
    @wraps(cls, updated=[])
    class SingletonWrapper(cls):
        _instance: Optional[T] = None
        
        def __new__(wrapped_cls, *args, **kwargs):
            if wrapped_cls._instance is None:
                with lock:
                    if wrapped_cls._instance is None:
                        wrapped_cls._instance = super().__new__(wrapped_cls)
            return wrapped_cls._instance
        
        @classmethod
        def get_instance(wrapped_cls) -> T:
            """获取单例实例"""
            if wrapped_cls._instance is None:
                wrapped_cls._instance = wrapped_cls()
            return wrapped_cls._instance
        
        @classmethod
        def reset_instance(wrapped_cls) -> None:
            """重置单例实例（用于测试）"""
            with lock:
                wrapped_cls._instance = None
    
    return SingletonWrapper


class LazySingleton:
    """
    懒加载单例基类
    
    使用方法:
        class MyService(LazySingleton):
            def _initialize(self):
                # 初始化逻辑
                pass
    """
    
    _instances: Dict[type, "LazySingleton"] = {}
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    def __init__(self, *args, **kwargs):
        if not self._initialized:
            self._initialize(*args, **kwargs)
            self._initialized = True
    
    def _initialize(self, *args, **kwargs) -> None:
        """子类重写此方法进行初始化"""
        pass
    
    @classmethod
    def get_instance(cls) -> "LazySingleton":
        """获取单例实例"""
        return cls()
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（用于测试）"""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]
            cls._initialized = False


class AsyncSingleton:
    """
    支持异步初始化的单例基类
    
    使用方法:
        class MyAsyncService(AsyncSingleton):
            async def _async_initialize(self):
                # 异步初始化逻辑
                await some_async_operation()
    """
    
    _instances: Dict[type, "AsyncSingleton"] = {}
    _lock: threading.Lock = threading.Lock()
    _async_initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[cls] = instance
        return cls._instances[cls]
    
    async def ensure_initialized(self) -> None:
        """确保异步初始化完成"""
        if not self._async_initialized:
            await self._async_initialize()
            self._async_initialized = True
    
    async def _async_initialize(self) -> None:
        """子类重写此方法进行异步初始化"""
        pass
    
    @classmethod
    def get_instance(cls) -> "AsyncSingleton":
        """获取单例实例"""
        return cls()
    
    @classmethod
    def reset_instance(cls) -> None:
        """重置单例实例（用于测试）"""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]
            cls._async_initialized = False
