"""
DeepSeek-OCR 客户端

基于 vLLM 部署的 DeepSeek-OCR 模型客户端
用于识别扫描版 PDF 和图片中的文字
"""
import base64
import io
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai 库未安装，OCR 功能不可用")

try:
    from pdf_summarizer.core.config import settings
except ImportError:
    settings = None


@dataclass
class OCRResult:
    """OCR 识别结果"""
    text: str
    page_num: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class OCRConfig:
    """OCR 配置"""
    api_base: str = "http://localhost:8001/v1"  # OCR 模型单独端口
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    api_key: str = "EMPTY"
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: float = 120.0


class DeepSeekOCRClient:
    """
    DeepSeek-OCR 客户端
    
    支持:
    - 单张图片 OCR
    - 多页 PDF OCR
    - 批量处理
    """
    
    _instance: Optional["DeepSeekOCRClient"] = None
    _shared_client: Optional[AsyncOpenAI] = None
    
    def __init__(self, config: Optional[OCRConfig] = None):
        if config:
            self.config = config
        else:
            # 从 settings 读取配置
            self.config = OCRConfig(
                api_base=getattr(settings, 'ocr_api_base', 'http://localhost:8001/v1'),
                model_name=getattr(settings, 'ocr_model_name', 'deepseek-ai/DeepSeek-OCR'),
            )
        
        self._client: Optional[AsyncOpenAI] = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls, config: Optional[OCRConfig] = None) -> "DeepSeekOCRClient":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def _ensure_client(self) -> AsyncOpenAI:
        """确保客户端已初始化"""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai 库未安装")
        
        if DeepSeekOCRClient._shared_client is None:
            DeepSeekOCRClient._shared_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout,
            )
        return DeepSeekOCRClient._shared_client
    
    @classmethod
    async def close_shared_client(cls) -> None:
        """关闭共享客户端"""
        if cls._shared_client:
            await cls._shared_client.close()
            cls._shared_client = None
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """将图片编码为 base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _get_image_mime_type(self, image_bytes: bytes) -> str:
        """检测图片 MIME 类型"""
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:4] == b'GIF8':
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        else:
            return "image/png"  # 默认
    
    async def ocr_image(
        self,
        image_bytes: bytes,
        prompt: str = "请识别图片中的所有文字内容，保持原有的格式和排版。",
    ) -> OCRResult:
        """
        识别单张图片中的文字
        
        Args:
            image_bytes: 图片字节数据
            prompt: 识别提示词
            
        Returns:
            OCR 识别结果
        """
        client = self._ensure_client()
        
        # 编码图片
        image_base64 = self._encode_image(image_bytes)
        mime_type = self._get_image_mime_type(image_bytes)
        
        try:
            response = await client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            text = response.choices[0].message.content.strip()
            
            return OCRResult(
                text=text,
                metadata={
                    "model": self.config.model_name,
                    "tokens": response.usage.total_tokens if response.usage else 0,
                }
            )
            
        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            raise
    
    async def ocr_pdf_pages(
        self,
        page_images: List[bytes],
        prompt: str = "请识别图片中的所有文字内容，保持原有的格式和排版。",
    ) -> List[OCRResult]:
        """
        批量识别 PDF 页面图片
        
        Args:
            page_images: 页面图片列表
            prompt: 识别提示词
            
        Returns:
            各页 OCR 结果列表
        """
        results = []
        total_pages = len(page_images)
        
        for i, image_bytes in enumerate(page_images):
            logger.info(f"OCR 识别第 {i+1}/{total_pages} 页...")
            
            try:
                result = await self.ocr_image(image_bytes, prompt)
                result.page_num = i + 1
                results.append(result)
            except Exception as e:
                logger.error(f"第 {i+1} 页 OCR 失败: {e}")
                results.append(OCRResult(
                    text="",
                    page_num=i + 1,
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            client = self._ensure_client()
            # 尝试获取模型列表
            await client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OCR 服务健康检查失败: {e}")
            return False


# 便捷函数
async def ocr_image(image_bytes: bytes) -> str:
    """快捷 OCR 函数"""
    client = DeepSeekOCRClient.get_instance()
    result = await client.ocr_image(image_bytes)
    return result.text
