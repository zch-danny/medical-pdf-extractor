"""
响应数据模型
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SummaryData(BaseModel):
    """摘要数据"""
    
    summary: str = Field(..., description="生成的摘要内容")
    word_count: int = Field(..., description="摘要字数/词数")
    language: str = Field(..., description="摘要语言")
    original_length: int = Field(..., description="原文长度（字符）")
    processing_time: float = Field(..., description="处理耗时（秒）")
    chunks_processed: int = Field(default=1, description="处理的文本块数")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "这是一段示例摘要内容...",
                    "word_count": 350,
                    "language": "zh",
                    "original_length": 15000,
                    "processing_time": 2.5,
                    "chunks_processed": 2
                }
            ]
        }
    }


class SummaryResponse(BaseModel):
    """摘要响应"""
    
    success: bool = Field(default=True, description="请求是否成功")
    data: Optional[SummaryData] = Field(default=None, description="摘要数据")
    message: str = Field(default="success", description="响应消息")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")


class ErrorResponse(BaseModel):
    """错误响应"""
    
    success: bool = Field(default=False)
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误详情")
    request_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "ValidationError",
                    "message": "文件大小超过限制",
                    "request_id": "abc123",
                    "timestamp": "2024-01-01T12:00:00"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """健康检查响应"""
    
    status: str = Field(default="healthy")
    version: str = Field(default="0.1.0")
    model: str = Field(default="Qwen3-8B")
    vllm_status: str = Field(default="unknown")
    timestamp: datetime = Field(default_factory=datetime.now)
