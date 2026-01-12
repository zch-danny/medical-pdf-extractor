"""Schemas module - 数据模型定义"""

from .request import SummarizeTextRequest, SummarizeFileRequest
from .response import SummaryResponse, ErrorResponse

__all__ = [
    "SummarizeTextRequest",
    "SummarizeFileRequest", 
    "SummaryResponse",
    "ErrorResponse",
]
