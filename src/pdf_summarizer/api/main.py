"""
FastAPI 应用主入口

增强功能:
- WebSocket 进度推送
- 导出 API
- 优雅关闭
"""

import signal
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from pdf_summarizer.core.config import settings, config_manager, reload_settings
from pdf_summarizer.models.llm_client import LLMClient
from pdf_summarizer.api.routes import router
from pdf_summarizer.api.async_routes import router as async_router
from pdf_summarizer.api.middleware import RequestLoggingMiddleware, ExceptionHandlerMiddleware
from pdf_summarizer.api.websocket import websocket_endpoint, manager as ws_manager
from pdf_summarizer.api.health import router as health_router, health_tracker
from pdf_summarizer.services.alert_service import AlertService
from pdf_summarizer.utils.exporters import export_summary, list_available_formats, SummaryExportData


# 全局关闭事件
shutdown_event: Optional[asyncio.Event] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（支持优雅关闭）"""
    global shutdown_event
    shutdown_event = asyncio.Event()
    
    # 启动时
    logger.info("PDF Summarization Service 启动中...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"vLLM API: {settings.vllm_api_base}")
    logger.info(f"WebSocket: enabled")
    
    # 标记服务就绪
    health_tracker.mark_ready()
    logger.info("健康检查端点: /health, /health/live, /health/ready")
    
    yield
    
    # 关闭时 - 优雅关闭
    logger.info("PDF Summarization Service 关闭中...")
    
    # 通知所有 WebSocket 客户端
    await ws_manager.broadcast_all({
        "type": "server_shutdown",
        "message": "服务即将关闭",
    })
    
    # 关闭 LLM 客户端
    await LLMClient.close_shared_client()
    
    # 关闭告警服务
    alert_service = AlertService.get_instance()
    await alert_service.close()
    
    logger.info("服务已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title="PDF Summarization Service",
        description="基于 Qwen3-8B 的 PDF 文档摘要服务",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # 中间件（按照洋葱模型，最先添加的最后执行）
    # 1. 异常处理（最外层）
    app.add_middleware(ExceptionHandlerMiddleware)
    
    # 2. 请求日志
    app.add_middleware(RequestLoggingMiddleware)
    
    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(health_router)  # 健康检查（无前缀）
    app.include_router(router, prefix="/api/v1")
    app.include_router(async_router, prefix="/api/v1")  # 异步任务路由
    
    # 根路由
    @app.get("/")
    async def root():
        return {
            "service": "PDF Summarization Service",
            "version": "0.1.0",
            "docs": "/docs",
            "features": [
                "PDF 摘要",
                "医学文献专用摘要",
                "WebSocket 进度推送",
                "多格式导出",
            ],
        }
    
    # WebSocket 端点
    @app.websocket("/ws/{client_id}")
    async def websocket_route(websocket: WebSocket, client_id: str):
        """
        WebSocket 连接端点
        
        支持的消息类型:
        - ping: 心跳
        - subscribe: 订阅任务进度 {"type": "subscribe", "task_id": "xxx"}
        - unsubscribe: 取消订阅
        """
        await websocket_endpoint(websocket, client_id)
    
    # WebSocket 状态
    @app.get("/ws/stats")
    async def ws_stats():
        """获取 WebSocket 连接统计"""
        return ws_manager.get_stats()
    
    # 导出格式列表
    @app.get("/api/v1/export/formats")
    async def get_export_formats():
        """获取支持的导出格式"""
        return {
            "success": True,
            "formats": list_available_formats(),
        }
    
    # 导出摘要
    @app.post("/api/v1/export")
    async def export_summary_api(
        summary: str,
        word_count: int,
        language: str = "zh",
        original_length: int = 0,
        processing_time: float = 0.0,
        chunks_processed: int = 1,
        style: str = "detailed",
        is_medical: bool = False,
        source_filename: str = None,
        format: str = Query(default="json", description="导出格式: json/markdown/pdf/docx"),
    ):
        """
        导出摘要结果
        
        支持格式: JSON, Markdown, PDF, DOCX
        """
        try:
            data = SummaryExportData(
                summary=summary,
                word_count=word_count,
                language=language,
                original_length=original_length,
                processing_time=processing_time,
                chunks_processed=chunks_processed,
                style=style,
                is_medical=is_medical,
                source_filename=source_filename,
            )
            
            content, content_type, extension = export_summary(data, format)
            
            filename = f"summary_{source_filename or 'export'}{extension}"
            
            return Response(
                content=content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename={filename}"
                }
            )
            
        except ValueError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return {"success": False, "error": f"导出失败: {e}"}
    
    # 配置管理 API
    @app.get("/api/v1/config")
    async def get_config():
        """获取当前运行时配置"""
        return {
            "success": True,
            "config": config_manager.get_runtime_config(),
        }
    
    @app.post("/api/v1/config/reload")
    async def reload_config():
        """
        热更新配置
        
        从 .env 文件重新加载配置（仅部分配置项支持热更新）
        """
        try:
            result = reload_settings()
            logger.info(f"配置已重新加载: {result['changes']}")
            return {
                "success": True,
                "message": "配置已重新加载",
                "reload_time": result["reload_time"],
                "changes": result["changes"],
            }
        except Exception as e:
            logger.error(f"配置重新加载失败: {e}")
            return {"success": False, "error": str(e)}
    
    return app


# 创建应用实例
app = create_app()
