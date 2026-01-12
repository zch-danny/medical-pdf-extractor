"""
OCR 增强的摘要服务

支持扫描版 PDF 的文字识别和摘要生成
"""
import time
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
from loguru import logger

from pdf_summarizer.core.config import settings
from pdf_summarizer.core.constants import MIN_TEXT_LENGTH
from pdf_summarizer.core.summary_styles import SummaryStyle
from pdf_summarizer.core.exceptions import InsufficientContentError
from pdf_summarizer.models.ocr_client import DeepSeekOCRClient, OCRResult
from pdf_summarizer.services.summarizer import (
    SummarizerService,
    SummaryResult,
    SummaryStrategy,
)
from pdf_summarizer.utils.pdf_to_images import (
    PDFToImages,
    check_pdf_has_text,
    pdf_to_images,
)
from pdf_summarizer.utils.pdf_parser import PDFParser


@dataclass
class OCRSummaryResult(SummaryResult):
    """OCR 摘要结果"""
    ocr_used: bool = False
    ocr_pages: int = 0
    text_coverage: float = 1.0


class OCRSummarizerService:
    """
    OCR 增强的摘要服务
    
    自动检测 PDF 类型并选择最优处理方式:
    - 文本型 PDF: 直接提取文本
    - 扫描版 PDF: 使用 DeepSeek-OCR 识别
    - 混合型 PDF: 结合两种方式
    """
    
    _instance: Optional["OCRSummarizerService"] = None
    
    def __init__(
        self,
        summarizer: Optional[SummarizerService] = None,
        ocr_client: Optional[DeepSeekOCRClient] = None,
        text_coverage_threshold: float = 0.3,  # 低于此值使用 OCR
    ):
        """
        初始化
        
        Args:
            summarizer: 摘要服务
            ocr_client: OCR 客户端
            text_coverage_threshold: 文本覆盖率阈值
        """
        self.summarizer = summarizer or SummarizerService()
        self.ocr_client = ocr_client  # 延迟初始化
        self.text_coverage_threshold = text_coverage_threshold
        self.pdf_parser = PDFParser(max_pages=settings.max_pdf_pages)
        self.pdf_to_images = PDFToImages(dpi=150)
        
        logger.info("OCRSummarizerService 初始化完成")
    
    @classmethod
    def get_instance(cls) -> "OCRSummarizerService":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _get_ocr_client(self) -> DeepSeekOCRClient:
        """获取 OCR 客户端（延迟初始化）"""
        if self.ocr_client is None:
            self.ocr_client = DeepSeekOCRClient.get_instance()
        return self.ocr_client
    
    async def summarize_pdf(
        self,
        pdf_bytes: bytes,
        language: Literal["zh", "en"] = "zh",
        max_length: int = 500,
        force_ocr: bool = False,
        style: SummaryStyle = "detailed",
        strategy: SummaryStrategy = "auto",
        is_medical: bool = False,
    ) -> OCRSummaryResult:
        """
        智能摘要 PDF
        
        自动检测是否需要 OCR
        
        Args:
            pdf_bytes: PDF 文件字节
            language: 输出语言
            max_length: 摘要最大长度
            force_ocr: 强制使用 OCR
            style: 摘要风格
            strategy: 摘要策略
            is_medical: 是否为医疗摘要
            
        Returns:
            摘要结果
        """
        start_time = time.time()
        
        # 1. 检测 PDF 是否有文本层
        has_text, text_coverage = check_pdf_has_text(pdf_bytes)
        
        logger.info(f"PDF 分析: has_text={has_text}, coverage={text_coverage:.0%}")
        
        # 2. 决定使用哪种方式
        use_ocr = force_ocr or (text_coverage < self.text_coverage_threshold)
        
        if use_ocr:
            logger.info("使用 OCR 模式处理 PDF")
            text, ocr_pages = await self._extract_with_ocr(pdf_bytes)
        else:
            logger.info("使用文本提取模式处理 PDF")
            text, _ = self.pdf_parser.extract_text_from_bytes(pdf_bytes)
            ocr_pages = 0
        
        # 3. 检查提取结果
        stripped = text.strip() if text else ""
        if len(stripped) < MIN_TEXT_LENGTH:
            # 文本提取失败，尝试 OCR
            if not use_ocr:
                logger.warning("文本提取失败，切换到 OCR 模式")
                text, ocr_pages = await self._extract_with_ocr(pdf_bytes)
                use_ocr = True
        
        stripped = text.strip() if text else ""
        if len(stripped) < MIN_TEXT_LENGTH:
            raise InsufficientContentError(
                message="PDF 内容为空或太短，无法生成摘要",
                min_chars=MIN_TEXT_LENGTH,
                actual_chars=len(stripped),
            )
        
        # 4. 生成摘要
        result = await self.summarizer.summarize_text(
            text=text,
            language=language,
            max_length=max_length,
            style=style,
            strategy=strategy,
            is_medical=is_medical,
        )
        
        processing_time = time.time() - start_time
        
        return OCRSummaryResult(
            summary=result.summary,
            word_count=result.word_count,
            language=result.language,
            original_length=result.original_length,
            processing_time=round(processing_time, 2),
            chunks_processed=result.chunks_processed,
            style=result.style,
            strategy=result.strategy,
            is_medical=result.is_medical,
            ocr_used=use_ocr,
            ocr_pages=ocr_pages,
            text_coverage=text_coverage,
        )
    
    async def _extract_with_ocr(
        self,
        pdf_bytes: bytes,
    ) -> Tuple[str, int]:
        """
        使用 OCR 提取 PDF 文字
        
        Args:
            pdf_bytes: PDF 字节
            
        Returns:
            (提取的文字, OCR处理的页数)
        """
        ocr_client = self._get_ocr_client()
        
        # 转换为图片
        page_images = self.pdf_to_images.convert_bytes(pdf_bytes)
        logger.info(f"PDF 转图片完成: {len(page_images)} 页")
        
        # OCR 识别
        image_bytes_list = [pi.image_bytes for pi in page_images]
        ocr_results = await ocr_client.ocr_pdf_pages(image_bytes_list)
        
        # 合并文字
        texts = []
        for result in ocr_results:
            if result.text:
                texts.append(f"=== 第 {result.page_num} 页 ===\n{result.text}")
        
        full_text = "\n\n".join(texts)
        
        logger.info(f"OCR 识别完成: {len(page_images)} 页, {len(full_text)} 字符")
        
        return full_text, len(page_images)
    
    async def health_check(self) -> dict:
        """健康检查"""
        ocr_ok = False
        try:
            ocr_client = self._get_ocr_client()
            ocr_ok = await ocr_client.health_check()
        except Exception as e:
            logger.warning(f"OCR 健康检查失败: {e}")
        
        return {
            "summarizer": "ok",
            "ocr": "ok" if ocr_ok else "unavailable",
        }
