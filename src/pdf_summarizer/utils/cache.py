"""
缓存模块

支持内存缓存和 Redis 缓存
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Optional, Any
from datetime import timedelta
from loguru import logger

from pdf_summarizer.core.config import settings


class CacheBackend(ABC):
    """缓存后端抽象基类"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """设置缓存"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass


class MemoryCache(CacheBackend):
    """内存缓存（LRU）"""
    
    def __init__(self, max_size: int = 1000):
        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl_map: dict = {}
    
    async def get(self, key: str) -> Optional[str]:
        import time
        if key in self._cache:
            # 检查是否过期
            if key in self._ttl_map and time.time() > self._ttl_map[key]:
                await self.delete(key)
                return None
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        import time
        # 如果达到最大容量，删除最老的
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest = next(iter(self._cache))
            await self.delete(oldest)
        
        self._cache[key] = value
        self._cache.move_to_end(key)
        self._ttl_map[key] = time.time() + ttl
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            self._ttl_map.pop(key, None)
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        return key in self._cache


class RedisCache(CacheBackend):
    """Redis 缓存"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "pdf_sum:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self._client = None
    
    async def _get_client(self):
        """懒加载 Redis 客户端"""
        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password or None,
                    decode_responses=True,
                )
                # 测试连接
                await self._client.ping()
                logger.info(f"Redis 连接成功: {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"Redis 连接失败: {e}，回退到内存缓存")
                self._client = None
                raise
        return self._client
    
    def _make_key(self, key: str) -> str:
        """添加前缀"""
        return f"{self.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[str]:
        try:
            client = await self._get_client()
            return await client.get(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis GET 失败: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        try:
            client = await self._get_client()
            await client.setex(self._make_key(key), ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis SET 失败: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            await client.delete(self._make_key(key))
            return True
        except Exception as e:
            logger.error(f"Redis DELETE 失败: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return await client.exists(self._make_key(key)) > 0
        except Exception:
            return False
    
    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.close()
            self._client = None


class SummaryCache:
    """摘要缓存服务"""
    
    _instance: Optional["SummaryCache"] = None
    
    def __init__(self):
        self.enabled = settings.cache_enabled
        self.ttl = settings.cache_ttl_seconds
        self._backend: Optional[CacheBackend] = None
    
    @classmethod
    def get_instance(cls) -> "SummaryCache":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def _get_backend(self) -> CacheBackend:
        """获取缓存后端"""
        if self._backend is None:
            # 尝试 Redis，失败则用内存
            try:
                self._backend = RedisCache(
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
                await self._backend._get_client()
            except Exception:
                logger.info("使用内存缓存")
                self._backend = MemoryCache()
        return self._backend
    
    def _generate_key(
        self,
        text: str,
        language: str,
        max_length: int,
        style: Optional[str] = None,
        is_medical: Optional[bool] = None,
        strategy: Optional[str] = None,
    ) -> str:
        """生成缓存键"""
        parts = [text, language, str(max_length)]
        if style is not None:
            parts.append(f"style={style}")
        if is_medical is not None:
            parts.append(f"is_medical={int(is_medical)}")
        if strategy is not None:
            parts.append(f"strategy={strategy}")

        content = ":".join(parts)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_summary(
        self,
        text: str,
        language: str,
        max_length: int,
        style: Optional[str] = None,
        is_medical: Optional[bool] = None,
        strategy: Optional[str] = None,
    ) -> Optional[dict]:
        """获取缓存的摘要"""
        if not self.enabled:
            return None
        
        try:
            backend = await self._get_backend()
            key = self._generate_key(
                text,
                language,
                max_length,
                style=style,
                is_medical=is_medical,
                strategy=strategy,
            )
            cached = await backend.get(key)
            
            if cached:
                logger.debug(f"缓存命中: {key[:8]}...")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None
    
    async def set_summary(
        self,
        text: str,
        language: str,
        max_length: int,
        summary_data: dict,
        style: Optional[str] = None,
        is_medical: Optional[bool] = None,
        strategy: Optional[str] = None,
    ) -> bool:
        """缓存摘要结果"""
        if not self.enabled:
            return False
        
        try:
            backend = await self._get_backend()
            key = self._generate_key(
                text,
                language,
                max_length,
                style=style,
                is_medical=is_medical,
                strategy=strategy,
            )
            value = json.dumps(summary_data, ensure_ascii=False)
            
            result = await backend.set(key, value, self.ttl)
            if result:
                logger.debug(f"缓存已保存: {key[:8]}...")
            return result
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
            return False
