"""
API 中间件

异常处理、请求日志、性能监控
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from pdf_summarizer.core.exceptions import InsufficientContentError


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 生成请求 ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = time.time()
        
        # 请求路径
        path = request.url.path
        method = request.method
        
        # 跳过健康检查的详细日志
        is_health_check = path.endswith("/health")
        
        if not is_health_check:
            logger.info(f"[{request_id}] {method} {path} - Started")
        
        try:
            response = await call_next(request)
            
            # 记录请求完成
            duration = time.time() - start_time
            status_code = response.status_code
            
            if not is_health_check:
                log_level = "INFO" if status_code < 400 else "WARNING"
                logger.log(
                    log_level,
                    f"[{request_id}] {method} {path} - {status_code} ({duration:.3f}s)"
                )
            
            # 添加请求 ID 到响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{request_id}] {method} {path} - Error: {e} ({duration:.3f}s)")
            raise


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常处理中间件"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        
        except InsufficientContentError as e:
            # 内容过短/不可处理
            logger.warning(f"InsufficientContentError: {e}")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "ContentTooShort",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )

        except ValueError as e:
            # 业务逻辑错误
            logger.warning(f"ValueError: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "ValidationError",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
        
        except ConnectionError as e:
            # 连接错误（如 vLLM 不可用）
            logger.error(f"ConnectionError: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "ServiceUnavailable",
                    "message": "推理服务暂时不可用，请稍后重试",
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
        
        except Exception as e:
            # 未知错误
            logger.exception(f"Unhandled exception: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "InternalServerError",
                    "message": "服务器内部错误",
                    "request_id": getattr(request.state, "request_id", None),
                }
            )
