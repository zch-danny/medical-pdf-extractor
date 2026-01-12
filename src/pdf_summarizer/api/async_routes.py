"""
异步任务 API 路由

支持:
- 异步提交 PDF 摘要任务
- 任务状态查询
- 任务结果获取
- 多种摘要风格
"""

import uuid
from typing import Literal, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from loguru import logger

from pdf_summarizer.core.config import settings
from pdf_summarizer.core.constants import MIN_TEXT_LENGTH
from pdf_summarizer.core.exceptions import InsufficientContentError
from pdf_summarizer.core.summary_styles import (
    SummaryStyle,
    get_style_config,
    list_styles,
)
from pdf_summarizer.services.task_manager import TaskManager, TaskInfo, TaskStatus
from pdf_summarizer.services.summarizer import SummarizerService
from pdf_summarizer.services.ocr_summarizer import OCRSummarizerService
from pdf_summarizer.utils.file_validator import validate_pdf_file
from pdf_summarizer.utils.pdf_parser import PDFParser
from pdf_summarizer.utils.pdf_to_images import check_pdf_has_text


router = APIRouter(prefix="/async", tags=["异步任务"])

# 服务实例
_summarizer_service: Optional[SummarizerService] = None


def get_summarizer_service() -> SummarizerService:
    """获取摘要服务"""
    global _summarizer_service
    if _summarizer_service is None:
        _summarizer_service = SummarizerService()
    return _summarizer_service


# ==================== 摘要风格 ====================

@router.get("/styles")
async def get_summary_styles():
    """
    获取支持的摘要风格列表
    
    返回所有可用的摘要风格及其描述
    """
    return {
        "success": True,
        "data": list_styles(),
    }


# ==================== 异步任务提交 ====================

@router.post("/summarize")
async def submit_summarize_task(
    file: UploadFile = File(..., description="PDF 文件"),
    language: Literal["zh", "en"] = Form(default="zh", description="输出语言"),
    style: Literal["brief", "detailed", "bullet", "academic", "medical"] = Form(
        default="detailed", description="摘要风格: brief/detailed/bullet/academic/medical(医学文献专用)"
    ),
    strategy: Literal["auto", "map_reduce", "refine"] = Form(
        default="auto", description="摘要策略: auto/map_reduce(快)/refine(质量高但慢)"
    ),
    enable_ocr: bool = Form(default=True, description="启用 OCR（扫描版 PDF）"),
    force_ocr: bool = Form(default=False, description="强制使用 OCR"),
):
    """
    提交异步 PDF 摘要任务（支持扫描版 PDF）
    
    - **file**: PDF 文件
    - **language**: 输出语言 (zh/en)
    - **style**: 摘要风格 (brief/detailed/bullet/academic)
    - **strategy**: 摘要策略 (auto/map_reduce/refine)
    - **enable_ocr**: 是否启用 OCR 识别扫描版 PDF
    - **force_ocr**: 强制使用 OCR（忽略文本层）
    
    返回任务ID，可通过 /task/{task_id} 查询状态
    """
    # 读取文件内容
    content = await file.read()
    
    # 文件安全校验
    validation = validate_pdf_file(content, file.filename, settings.max_pdf_size_mb)
    
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.error)
    
    # 获取风格配置
    style_config = get_style_config(style)
    
    # 检测 PDF 是否需要 OCR
    has_text, text_coverage = check_pdf_has_text(content)
    need_ocr = enable_ocr and (force_ocr or text_coverage < settings.text_coverage_threshold)
    
    # 定义任务函数
    async def process_task(task_info: TaskInfo) -> dict:
        """处理摘要任务"""
        task_manager = TaskManager.get_instance()
        
        try:
            if need_ocr:
                # 使用 OCR 模式
                task_manager.update_progress(task_info.task_id, 10, "正在进行 OCR 识别...")
                
                ocr_service = OCRSummarizerService.get_instance()
                result = await ocr_service.summarize_pdf(
                    pdf_bytes=content,
                    language=language,
                    max_length=style_config.max_length,
                    force_ocr=force_ocr,
                    style=style,
                    strategy=strategy,
                )
                
                task_manager.update_progress(task_info.task_id, 100, "摘要生成完成")
                
                return {
                    "summary": result.summary,
                    "word_count": result.word_count,
                    "language": result.language,
                    "style": style,
                    "style_name": style_config.name,
                    "strategy": result.strategy,
                    "original_length": result.original_length,
                    "processing_time": result.processing_time,
                    "chunks_processed": result.chunks_processed,
                    "ocr_used": result.ocr_used,
                    "ocr_pages": result.ocr_pages,
                    "text_coverage": result.text_coverage,
                    "file_validation": validation.to_dict(),
                }
            else:
                # 使用文本提取模式
                task_manager.update_progress(task_info.task_id, 10, "正在解析 PDF...")
                
                pdf_parser = PDFParser(max_pages=settings.max_pdf_pages)
                text, metadata = pdf_parser.extract_text_from_bytes(content)
                
                stripped = text.strip() if text else ""
                if len(stripped) < MIN_TEXT_LENGTH:
                    raise InsufficientContentError(
                        message="PDF 内容为空或太短，无法生成摘要",
                        min_chars=MIN_TEXT_LENGTH,
                        actual_chars=len(stripped),
                    )
                
                task_manager.update_progress(task_info.task_id, 30, "正在生成摘要...")
                
                service = get_summarizer_service()
                result = await service.summarize_text(
                    text=text,
                    language=language,
                    max_length=style_config.max_length,
                    strategy=strategy,
                    style=style,
                )
                
                task_manager.update_progress(task_info.task_id, 100, "摘要生成完成")
                
                return {
                    "summary": result.summary,
                    "word_count": result.word_count,
                    "language": result.language,
                    "style": style,
                    "style_name": style_config.name,
                    "strategy": result.strategy,
                    "original_length": result.original_length,
                    "processing_time": result.processing_time,
                    "chunks_processed": result.chunks_processed,
                    "ocr_used": False,
                    "ocr_pages": 0,
                    "text_coverage": text_coverage,
                    "pdf_metadata": metadata,
                    "file_validation": validation.to_dict(),
                }
            
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
            raise
    
    # 提交任务
    task_manager = TaskManager.get_instance()
    task_id = await task_manager.submit(
        task_fn=process_task,
        metadata={
            "filename": file.filename,
            "language": language,
            "style": style,
            "strategy": strategy,
            "file_size": len(content),
            "ocr_enabled": enable_ocr,
            "need_ocr": need_ocr,
            "text_coverage": text_coverage,
        }
    )
    
    logger.info(f"异步任务已提交: {task_id}, 文件: {file.filename}, OCR: {need_ocr}")
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "任务已提交，请通过 /task/{task_id} 查询状态",
        "style": style,
        "style_name": style_config.name,
        "strategy": strategy,
        "ocr_mode": need_ocr,
        "text_coverage": f"{text_coverage:.0%}",
    }


# ==================== 任务状态查询 ====================

@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    查询任务状态
    
    - **task_id**: 任务ID
    
    返回任务当前状态、进度和结果（如已完成）
    """
    task_manager = TaskManager.get_instance()
    task_info = task_manager.get_task(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    
    response = {
        "success": True,
        "task": task_info.to_dict(),
    }
    
    # 如果任务完成，包含结果
    if task_info.status == TaskStatus.COMPLETED and task_info.result:
        response["data"] = task_info.result
    
    return response


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="按状态筛选"),
    limit: int = Query(20, ge=1, le=100, description="返回数量"),
):
    """
    列出最近的任务
    
    - **status**: 可选，按状态筛选 (pending/processing/completed/failed)
    - **limit**: 返回数量限制
    """
    task_manager = TaskManager.get_instance()
    
    tasks = []
    for task_id, task_info in list(task_manager._tasks.items())[-limit:]:
        if status and task_info.status.value != status:
            continue
        tasks.append(task_info.to_dict())
    
    return {
        "success": True,
        "count": len(tasks),
        "tasks": tasks,
        "stats": task_manager.get_stats(),
    }


# ==================== 文件验证 ====================

@router.post("/validate")
async def validate_file(
    file: UploadFile = File(..., description="要验证的文件"),
):
    """
    验证 PDF 文件
    
    检查文件格式、大小、安全性等
    """
    content = await file.read()
    
    validation = validate_pdf_file(
        content,
        file.filename,
        settings.max_pdf_size_mb,
    )
    
    return {
        "success": True,
        "validation": validation.to_dict(),
    }
