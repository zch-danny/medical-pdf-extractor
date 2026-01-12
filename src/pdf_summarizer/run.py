"""
应用启动入口
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from loguru import logger

from pdf_summarizer.core.config import settings


def setup_logging():
    """配置日志"""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )
    
    # 添加文件输出
    logger.add(
        log_dir / "pdf_summarizer_{time:YYYY-MM-DD}.log",
        level=settings.log_level,
        rotation="00:00",
        retention="7 days",
        compression="gz",
        encoding="utf-8",
    )


def main():
    """主函数"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("PDF Summarization Service")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"vLLM API: {settings.vllm_api_base}")
    logger.info(f"API: http://{settings.api_host}:{settings.api_port}")
    logger.info("=" * 60)
    
    uvicorn.run(
        "pdf_summarizer.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if settings.app_env == "prod" else 1,
        reload=settings.api_debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
