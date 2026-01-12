"""
API 路由定义
"""

import time
import uuid
from typing import Literal
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, Request
from loguru import logger

from pdf_summarizer.core.config import settings
from pdf_summarizer.core.exceptions import InsufficientContentError
from pdf_summarizer.schemas.request import SummarizeTextRequest
from pdf_summarizer.schemas.response import (
    SummaryResponse,
    SummaryData,
    ErrorResponse,
    HealthResponse,
)
from pdf_summarizer.services.summarizer import SummarizerService
from pdf_summarizer.models.llm_client import LLMClient
from pdf_summarizer.api.rate_limit import check_rate_limit, RateLimiter
from pdf_summarizer.services.backpressure import BackpressureController, BackpressureContext
from pdf_summarizer.services.alert_service import AlertService
from pdf_summarizer.utils.tiered_cache import TieredCache


router = APIRouter()

# 服务实例（懒加载）
_summarizer_service: SummarizerService = None


def get_summarizer_service() -> SummarizerService:
    """获取摘要服务实例"""
    global _summarizer_service
    if _summarizer_service is None:
        _summarizer_service = SummarizerService()
    return _summarizer_service


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    llm_client = LLMClient()
    vllm_ok = await llm_client.health_check()
    
    return HealthResponse(
        status="healthy" if vllm_ok else "degraded",
        version="0.1.0",
        model=settings.model_name,
        vllm_status="connected" if vllm_ok else "disconnected",
    )


@router.post("/summarize/text", response_model=SummaryResponse)
async def summarize_text(
    request: SummarizeTextRequest,
    _: bool = Depends(check_rate_limit),
):
    """
    对文本进行摘要
    
    - **text**: 需要摘要的文本（最少100字符）
    - **language**: 输出语言 (zh/en)
    - **max_length**: 摘要最大长度
    """
    request_id = str(uuid.uuid4())[:8]
    
    try:
        service = get_summarizer_service()
        
        result = await service.summarize_text(
            text=request.text,
            language=request.language,
            max_length=request.max_length,
        )
        
        return SummaryResponse(
            success=True,
            data=SummaryData(
                summary=result.summary,
                word_count=result.word_count,
                language=result.language,
                original_length=result.original_length,
                processing_time=result.processing_time,
                chunks_processed=result.chunks_processed,
            ),
            message="摘要生成成功",
            request_id=request_id,
        )
        
    except InsufficientContentError as e:
        logger.warning(f"[{request_id}] 内容过短: {e}")
        raise

    except ValueError as e:
        logger.warning(f"[{request_id}] 参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"[{request_id}] 摘要生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"摘要生成失败: {e}")


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_pdf(
    file: UploadFile = File(..., description="PDF 文件"),
    language: Literal["zh", "en"] = Form(default="zh", description="输出语言"),
    max_length: int = Form(default=500, ge=100, le=2000, description="摘要最大长度"),
    _: bool = Depends(check_rate_limit),
):
    """
    上传 PDF 文件并生成摘要
    
    - **file**: PDF 文件
    - **language**: 输出语言 (zh/en)
    - **max_length**: 摘要最大长度
    """
    request_id = str(uuid.uuid4())[:8]
    
    # 验证文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    
    # 验证文件大小
    content = await file.read()
    if len(content) > settings.max_pdf_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制 ({settings.max_pdf_size_mb}MB)"
        )
    
    try:
        logger.info(f"[{request_id}] 开始处理 PDF: {file.filename}")
        
        service = get_summarizer_service()
        
        result = await service.summarize_pdf(
            pdf_bytes=content,
            language=language,
            max_length=max_length,
        )
        
        logger.info(f"[{request_id}] PDF 摘要完成: {file.filename}")
        
        return SummaryResponse(
            success=True,
            data=SummaryData(
                summary=result.summary,
                word_count=result.word_count,
                language=result.language,
                original_length=result.original_length,
                processing_time=result.processing_time,
                chunks_processed=result.chunks_processed,
            ),
            message="PDF 摘要生成成功",
            request_id=request_id,
        )
        
    except InsufficientContentError as e:
        logger.warning(f"[{request_id}] 内容过短: {e}")
        raise

    except ValueError as e:
        logger.warning(f"[{request_id}] PDF 处理错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"[{request_id}] PDF 摘要生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"PDF 摘要生成失败: {e}")


# ==================== 统计接口 ====================

@router.get("/stats")
async def get_stats():
    """
    获取服务统计信息
    
    包含：缓存统计、背压统计、告警指标、限流状态
    """
    stats = {
        "service": "PDF Summarization Service",
        "version": "0.1.0",
    }
    
    # 缓存统计
    try:
        cache = TieredCache.get_instance()
        stats["cache"] = cache.get_stats()
    except Exception as e:
        stats["cache"] = {"error": str(e)}
    
    # 背压统计
    try:
        backpressure = BackpressureController.get_instance()
        stats["backpressure"] = backpressure.get_stats()
    except Exception as e:
        stats["backpressure"] = {"error": str(e)}
    
    # 告警指标
    try:
        alert_service = AlertService.get_instance()
        stats["metrics"] = alert_service.get_metrics()
    except Exception as e:
        stats["metrics"] = {"error": str(e)}
    
    # 限流状态
    try:
        limiter = RateLimiter.get_instance()
        stats["rate_limit"] = {
            "enabled": limiter.config.enabled,
            "requests_per_minute": limiter.config.requests_per_minute,
            "requests_per_hour": limiter.config.requests_per_hour,
        }
    except Exception as e:
        stats["rate_limit"] = {"error": str(e)}
    
    return stats


@router.get("/stats/cache")
async def get_cache_stats():
    """获取缓存详细统计"""
    try:
        cache = TieredCache.get_instance()
        return {"success": True, "data": cache.get_stats()}
    except Exception as e:
        return {"success": False, "error": str(e)}
