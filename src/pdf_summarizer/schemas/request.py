"""
请求数据模型
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class SummarizeTextRequest(BaseModel):
    """文本摘要请求"""
    
    text: str = Field(
        ...,
        min_length=100,
        description="需要摘要的文本内容，最少100字符"
    )
    language: Literal["zh", "en"] = Field(
        default="zh",
        description="输出摘要的语言：zh(中文) 或 en(英文)"
    )
    max_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="摘要最大长度（字/词）"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "这是一段需要摘要的长文本...",
                    "language": "zh",
                    "max_length": 500
                }
            ]
        }
    }


class SummarizeFileRequest(BaseModel):
    """文件摘要请求参数（用于表单）"""
    
    language: Literal["zh", "en"] = Field(
        default="zh",
        description="输出摘要的语言"
    )
    max_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="摘要最大长度"
    )
    extract_images: bool = Field(
        default=False,
        description="是否提取图片信息（暂未实现）"
    )
