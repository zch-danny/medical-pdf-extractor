"""
结构化日志模块

支持:
- JSON 结构化日志输出
- 请求链路追踪 ID
- 分级日志输出
- 日志上下文管理
"""

import sys
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
from loguru import logger

from pdf_summarizer.core.config import settings


# 请求上下文变量
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
trace_context_var: ContextVar[Dict[str, Any]] = ContextVar("trace_context", default={})


def get_request_id() -> Optional[str]:
    """获取当前请求 ID"""
    return request_id_var.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    设置当前请求 ID
    
    Args:
        request_id: 请求 ID，如果为 None 则自动生成
        
    Returns:
        请求 ID
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    request_id_var.set(request_id)
    return request_id


def get_trace_context() -> Dict[str, Any]:
    """获取追踪上下文"""
    return trace_context_var.get()


def set_trace_context(context: Dict[str, Any]) -> None:
    """设置追踪上下文"""
    trace_context_var.set(context)


def add_trace_context(**kwargs) -> None:
    """添加追踪上下文字段"""
    current = trace_context_var.get().copy()
    current.update(kwargs)
    trace_context_var.set(current)


def json_serializer(record: Dict[str, Any]) -> str:
    """
    JSON 日志序列化器
    
    输出 JSON 格式的日志，便于日志聚合和分析
    """
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name,
        "service": "pdf_summarizer",
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # 添加请求 ID
    request_id = get_request_id()
    if request_id:
        log_entry["request_id"] = request_id
    
    # 添加追踪上下文
    trace_ctx = get_trace_context()
    if trace_ctx:
        log_entry["trace"] = trace_ctx
    
    # 添加异常信息
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
        }
    
    # 添加额外字段
    if record.get("extra"):
        for k, v in record["extra"].items():
            if k not in ["request_id"]:
                log_entry[k] = v
    
    return json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"


def console_format(record: Dict[str, Any]) -> str:
    """控制台日志格式"""
    request_id = get_request_id()
    req_id_str = f"<cyan>{request_id}</cyan> | " if request_id else ""
    
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"{req_id_str}"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>\n"
    )


def setup_logging(
    log_level: str = None,
    log_dir: str = None,
    json_output: bool = False,
) -> logger:
    """
    配置日志系统
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        json_output: 是否输出 JSON 格式（生产环境建议开启）
    """
    log_level = log_level or settings.log_level
    log_dir = log_dir or settings.log_dir
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台输出
    if json_output:
        logger.add(
            sys.stderr,
            level=log_level,
            format=json_serializer,
            colorize=False,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True,
            enqueue=True,
        )
    
    # 文件输出 - 应用日志（JSON 格式）
    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        level=log_level,
        format=json_serializer,
        rotation="100 MB",  # 100MB or daily, whichever comes first
        retention="7 days",
        compression="gz",
        encoding="utf-8",
        enqueue=True,
    )
    
    # 文件输出 - 错误日志（JSON 格式）
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format=json_serializer,
        rotation="100 MB",  # 100MB or daily, whichever comes first
        retention="30 days",
        compression="gz",
        encoding="utf-8",
        enqueue=True,
    )
    
    logger.info(f"日志系统初始化完成: level={log_level}, dir={log_dir}")
    return logger


def log_with_context(level: str, message: str, **kwargs) -> None:
    """
    带上下文的日志记录
    
    Args:
        level: 日志级别
        message: 日志消息
        **kwargs: 额外字段
    """
    log_func = getattr(logger, level.lower(), logger.info)
    context = get_trace_context().copy()
    context.update(kwargs)
    
    with logger.contextualize(**context):
        log_func(message)


def log_operation(
    operation: str,
    success: bool = True,
    duration_ms: Optional[float] = None,
    **kwargs
) -> None:
    """
    记录操作日志
    
    Args:
        operation: 操作名称
        success: 是否成功
        duration_ms: 耗时（毫秒）
        **kwargs: 额外字段
    """
    level = "INFO" if success else "ERROR"
    log_data = {"operation": operation, "success": success}
    
    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)
    
    log_data.update(kwargs)
    log_with_context(level, f"Operation: {operation}", **log_data)


class LogContext:
    """
    日志上下文管理器
    
    使用方法:
        with LogContext(operation="summarize", user_id="123"):
            logger.info("Processing...")
    """
    
    def __init__(self, **kwargs):
        self.context = kwargs
        self._previous_context = None
    
    def __enter__(self):
        self._previous_context = get_trace_context()
        new_context = self._previous_context.copy()
        new_context.update(self.context)
        set_trace_context(new_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_trace_context(self._previous_context or {})
        return False


def trace_operation(operation: str):
    """
    操作追踪装饰器
    
    使用方法:
        @trace_operation("summarize_pdf")
        async def summarize_pdf(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            with LogContext(operation=operation):
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    log_operation(operation, success=True, duration_ms=duration_ms)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    log_operation(operation, success=False, duration_ms=duration_ms, error=str(e))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            with LogContext(operation=operation):
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    log_operation(operation, success=True, duration_ms=duration_ms)
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    log_operation(operation, success=False, duration_ms=duration_ms, error=str(e))
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
