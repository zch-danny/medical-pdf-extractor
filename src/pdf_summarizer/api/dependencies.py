"""
API 依赖注入

提供服务实例、认证等
"""

from typing import Optional
from fastapi import Depends, HTTPException, Header
from loguru import logger

from pdf_summarizer.core.config import settings
from pdf_summarizer.services.summarizer import SummarizerService


# 服务单例
_summarizer_service: Optional[SummarizerService] = None


def get_summarizer_service() -> SummarizerService:
    """获取摘要服务实例（单例）"""
    global _summarizer_service
    if _summarizer_service is None:
        _summarizer_service = SummarizerService()
        logger.info("SummarizerService 单例已创建")
    return _summarizer_service


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> bool:
    """
    可选的 API Key 验证
    
    在 .env 中设置 API_KEY 启用验证
    """
    api_key = getattr(settings, 'api_key', None)
    
    # 如果未配置 API Key，跳过验证
    if not api_key:
        return True
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if x_api_key != api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key",
        )
    
    return True
