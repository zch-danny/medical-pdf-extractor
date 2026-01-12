"""
分层缓存服务 - 高性能版本

架构:
- L1: 进程内 LRU 缓存 (<0.1ms)
- L2: Redis 缓存 (<5ms)

参考 safety_llm_deploy 的缓存架构
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock
from loguru import logger

try:
    from pdf_summarizer.core.high_performance import TIERED_CACHE_CONFIG
    HIGH_PERF_AVAILABLE = True
except ImportError:
    HIGH_PERF_AVAILABLE = False
    TIERED_CACHE_CONFIG = {
        "l1": {"enabled": True, "max_size": 500, "ttl_seconds": 600},
        "l2": {"enabled": False},
    }


# ==================== 缓存统计 ====================

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


# ==================== 缓存条目 ====================

@dataclass
class CacheEntry:
    """缓存条目"""
    data: Dict[str, Any]
    created_at: float
    hit_count: int = 0


# ==================== L1 内存缓存 ====================

class L1Cache:
    """
    L1 进程内 LRU 缓存
    
    特点:
    - 最快 (<0.1ms)
    - 进程隔离
    - 线程安全
    """
    
    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 600,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._stats = CacheStats()
        
        logger.info(f"L1Cache 初始化: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _compute_key(self, text: str, language: str, max_length: int) -> str:
        """计算缓存键"""
        content = f"{text}:{language}:{max_length}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, language: str, max_length: int) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        key = self._compute_key(text, language, max_length)
        
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if time.time() - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._stats.misses += 1
                return None
            
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats.hits += 1
            
            return entry.data
    
    def set(self, text: str, language: str, max_length: int, data: Dict[str, Any]) -> None:
        """设置缓存"""
        key = self._compute_key(text, language, max_length)
        
        with self._lock:
            # 淘汰过期和超量条目
            self._evict_if_needed()
            
            self._cache[key] = CacheEntry(
                data=data,
                created_at=time.time(),
            )
            self._cache.move_to_end(key)
            self._stats.size = len(self._cache)
    
    def _evict_if_needed(self) -> None:
        """淘汰策略"""
        now = time.time()
        
        # 淘汰过期条目
        expired = [
            k for k, v in self._cache.items()
            if now - v.created_at > self.ttl_seconds
        ]
        for k in expired:
            del self._cache[k]
            self._stats.evictions += 1
        
        # 淘汰超量条目 (LRU)
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1
    
    def clear(self) -> int:
        """清空缓存"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            self._stats.size = len(self._cache)
            return {"l1": self._stats.to_dict()}


# ==================== L2 Redis 缓存 ====================

class L2Cache:
    """
    L2 Redis 缓存
    
    特点:
    - 较快 (<5ms)
    - 跨进程共享
    - 持久化
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "pdf_sum:",
        ttl_seconds: int = 3600,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._client = None
        self._available = False
        self._stats = CacheStats()
        
        self._try_connect()
    
    def _try_connect(self) -> None:
        """尝试连接 Redis"""
        try:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            self._available = True
            logger.info(f"L2Cache (Redis) 连接成功: {self.host}:{self.port}")
        except Exception as e:
            self._available = False
            logger.warning(f"L2Cache (Redis) 连接失败: {e}，将仅使用 L1 缓存")
    
    def _make_key(self, text: str, language: str, max_length: int) -> str:
        """生成 Redis 键"""
        content = f"{text}:{language}:{max_length}"
        hash_key = hashlib.md5(content.encode()).hexdigest()
        return f"{self.key_prefix}{hash_key}"
    
    def get(self, text: str, language: str, max_length: int) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        if not self._available:
            return None
        
        try:
            key = self._make_key(text, language, max_length)
            data = self._client.get(key)
            
            if data:
                self._stats.hits += 1
                return json.loads(data)
            
            self._stats.misses += 1
            return None
        except Exception as e:
            logger.warning(f"L2 GET 失败: {e}")
            return None
    
    def set(self, text: str, language: str, max_length: int, data: Dict[str, Any]) -> bool:
        """设置缓存"""
        if not self._available:
            return False
        
        try:
            key = self._make_key(text, language, max_length)
            self._client.setex(key, self.ttl_seconds, json.dumps(data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning(f"L2 SET 失败: {e}")
            return False
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {"l2": self._stats.to_dict(), "available": self._available}


# ==================== 分层缓存管理器 ====================

class TieredCache:
    """
    分层缓存管理器
    
    查询流程: L1 -> L2 -> Miss
    写入流程: L1 + L2
    """
    
    _instance: Optional["TieredCache"] = None
    
    def __init__(self):
        config = TIERED_CACHE_CONFIG
        
        # 初始化 L1
        l1_config = config.get("l1", {})
        self.l1_enabled = l1_config.get("enabled", True)
        self.l1 = L1Cache(
            max_size=l1_config.get("max_size", 500),
            ttl_seconds=l1_config.get("ttl_seconds", 600),
        ) if self.l1_enabled else None
        
        # 初始化 L2
        l2_config = config.get("l2", {})
        self.l2_enabled = l2_config.get("enabled", False)
        self.l2 = L2Cache(
            host=l2_config.get("host", "localhost"),
            port=l2_config.get("port", 6379),
            db=l2_config.get("db", 0),
            password=l2_config.get("password"),
            key_prefix=l2_config.get("key_prefix", "pdf_sum:"),
            ttl_seconds=l2_config.get("ttl_seconds", 3600),
        ) if self.l2_enabled else None
        
        logger.info(f"TieredCache 初始化: L1={self.l1_enabled}, L2={self.l2_enabled}")
    
    @classmethod
    def get_instance(cls) -> "TieredCache":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get(self, text: str, language: str, max_length: int) -> Optional[Dict[str, Any]]:
        """
        分层查询
        
        流程: L1 -> L2 -> Miss
        """
        # L1 查询
        if self.l1:
            data = self.l1.get(text, language, max_length)
            if data:
                logger.debug("L1 缓存命中")
                return data
        
        # L2 查询
        if self.l2:
            data = self.l2.get(text, language, max_length)
            if data:
                logger.debug("L2 缓存命中")
                # 回填 L1
                if self.l1:
                    self.l1.set(text, language, max_length, data)
                return data
        
        return None
    
    def set(self, text: str, language: str, max_length: int, data: Dict[str, Any]) -> None:
        """
        写入所有层
        """
        if self.l1:
            self.l1.set(text, language, max_length, data)
        
        if self.l2:
            self.l2.set(text, language, max_length, data)
    
    def get_stats(self) -> dict:
        """获取所有层统计"""
        stats = {}
        if self.l1:
            stats.update(self.l1.get_stats())
        if self.l2:
            stats.update(self.l2.get_stats())
        return stats
