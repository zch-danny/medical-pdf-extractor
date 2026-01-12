"""
LLM 客户端

通过 OpenAI 兼容 API 调用 vLLM 服务

增强功能:
- 指数退避重试
- 熔断器集成
- 超时控制
- 连接池优化
"""

import asyncio
from typing import Optional, List, Dict, Any
from loguru import logger
import httpx

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        RetryError,
    )
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("tenacity 库未安装，重试功能受限")

try:
    from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai 库未安装，LLM 客户端不可用")
    # 定义占位异常类
    class APIError(Exception): pass
    class APIConnectionError(Exception): pass
    class RateLimitError(Exception): pass
    class APITimeoutError(Exception): pass

from pdf_summarizer.core.config import settings
from pdf_summarizer.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    get_llm_circuit_breaker,
)


# 可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    asyncio.TimeoutError,
)

# 不可重试的异常类型
NON_RETRYABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
)


class LLMClient:
    """
    LLM 客户端
    
    使用 OpenAI 兼容 API 调用 vLLM 推理服务
    
    增强功能:
    - 指数退避重试
    - 熔断器集成
    - 超时控制
    """
    
    _shared_client: Optional[AsyncOpenAI] = None
    _shared_http_client: Optional[httpx.AsyncClient] = None
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_circuit_breaker: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_base: vLLM API 地址
            api_key: API 密钥
            model_name: 模型名称
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            enable_circuit_breaker: 是否启用熔断器
            max_retries: 最大重试次数
            retry_delay: 重试基础延迟（秒）
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装 openai: pip install openai")
        
        self.api_base = api_base or settings.vllm_api_base
        self.api_key = api_key or settings.vllm_api_key
        self.model_name = model_name or settings.model_name
        self.max_tokens = max_tokens or settings.model_max_tokens
        self.temperature = temperature or settings.model_temperature
        
        self.enable_circuit_breaker = enable_circuit_breaker
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client: Optional[AsyncOpenAI] = None
        self._initialized = False
        self._circuit_breaker: Optional[CircuitBreaker] = None
    
    @classmethod
    def _get_shared_client(cls, api_base: str, api_key: str) -> AsyncOpenAI:
        """获取共享客户端（连接池优化）"""
        if cls._shared_client is None:
            # 配置 HTTP 客户端
            timeout = httpx.Timeout(
                connect=10.0,
                read=120.0,  # 生成可能较慢
                write=10.0,
                pool=10.0
            )
            limits = httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            )
            
            cls._shared_http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=False,
            )
            
            cls._shared_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                http_client=cls._shared_http_client,
                max_retries=2,
            )
            
            logger.info(f"LLM 客户端初始化完成: {api_base}")
        
        return cls._shared_client
    
    @classmethod
    async def close_shared_client(cls):
        """关闭共享客户端"""
        if cls._shared_http_client:
            await cls._shared_http_client.aclose()
            cls._shared_http_client = None
            cls._shared_client = None
            logger.info("LLM 客户端已关闭")
    
    def initialize(self):
        """初始化客户端"""
        if not self._initialized:
            self.client = self._get_shared_client(self.api_base, self.api_key)
            if self.enable_circuit_breaker:
                self._circuit_breaker = get_llm_circuit_breaker()
            self._initialized = True
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """
        生成文本（带重试和熔断器）
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            stop: 停止词列表
            timeout: 超时时间（秒）
            
        Returns:
            生成的文本
        """
        if not self._initialized:
            self.initialize()
        
        # 检查熔断器状态
        if self._circuit_breaker and self._circuit_breaker.is_open:
            raise CircuitBreakerOpenError("LLM 服务熔断中，请稍后重试")
        
        try:
            result = await self._generate_with_retry(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                timeout=timeout,
            )
            
            # 记录成功
            if self._circuit_breaker:
                await self._circuit_breaker.record_success()
            
            return result
            
        except Exception as e:
            # 记录失败
            if self._circuit_breaker:
                await self._circuit_breaker.record_failure(e)
            raise
    
    async def _generate_with_retry(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """带重试的生成方法"""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._do_generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    timeout=timeout,
                )
                return result
                
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(
                        f"LLM 请求失败 (attempt {attempt + 1}/{self.max_retries + 1}): {e}, "
                        f"将在 {wait_time:.1f}s 后重试"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"LLM 请求失败，已耗尽重试次数: {e}")
                    
            except NON_RETRYABLE_EXCEPTIONS as e:
                logger.error(f"LLM 请求失败（不可重试）: {e}")
                raise
                
            except Exception as e:
                last_error = e
                logger.error(f"LLM 请求未知错误: {e}")
                if attempt >= self.max_retries:
                    raise
        
        raise RuntimeError(f"LLM 生成失败: {last_error}") from last_error
    
    async def _do_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """实际执行生成"""
        try:
            # 应用超时
            if timeout:
                response = await asyncio.wait_for(
                    self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        max_tokens=max_tokens or self.max_tokens,
                        temperature=temperature or self.temperature,
                        stop=stop,
                    ),
                    timeout=timeout
                )
            else:
                response = await self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    stop=stop,
                )
            
            generated_text = response.choices[0].text.strip()
            
            logger.debug(
                f"LLM 生成完成: prompt_len={len(prompt)}, "
                f"output_len={len(generated_text)}"
            )
            
            return generated_text
            
        except asyncio.TimeoutError:
            raise APITimeoutError(f"LLM 请求超时: timeout={timeout}s")
        except Exception as e:
            raise
    
    async def chat(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        对话模式生成（Chat Completions API）
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        if not self._initialized:
            self.initialize()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"LLM Chat 生成失败: {e}")
            raise RuntimeError(f"LLM Chat 生成失败: {e}") from e
    
    async def health_check(self) -> bool:
        """检查 vLLM 服务是否可用"""
        try:
            if not self._initialized:
                self.initialize()
            
            # 尝试获取模型列表
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.warning(f"vLLM 健康检查失败: {e}")
            return False
