"""
PDF 转图片工具

将 PDF 页面转换为图片，用于 OCR 识别
"""
import io
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF 未安装，PDF 转图片功能不可用")


@dataclass
class PageImage:
    """页面图片"""
    page_num: int
    image_bytes: bytes
    width: int
    height: int
    dpi: int


class PDFToImages:
    """
    PDF 转图片工具
    
    使用 PyMuPDF 将 PDF 页面渲染为图片
    """
    
    def __init__(
        self,
        dpi: int = 150,
        max_pages: int = 100,
        image_format: str = "png",
    ):
        """
        初始化
        
        Args:
            dpi: 渲染 DPI（越高越清晰，但越慢）
            max_pages: 最大处理页数
            image_format: 输出图片格式 (png/jpeg)
        """
        self.dpi = dpi
        self.max_pages = max_pages
        self.image_format = image_format
        self.zoom = dpi / 72.0  # PDF 默认 72 DPI
    
    def convert_bytes(self, pdf_bytes: bytes) -> List[PageImage]:
        """
        将 PDF 字节转换为图片列表
        
        Args:
            pdf_bytes: PDF 文件字节
            
        Returns:
            页面图片列表
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF 未安装")
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_images = []
        
        try:
            total_pages = min(len(doc), self.max_pages)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                
                # 渲染页面为图片
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为字节
                if self.image_format == "jpeg":
                    image_bytes = pix.tobytes("jpeg")
                else:
                    image_bytes = pix.tobytes("png")
                
                page_images.append(PageImage(
                    page_num=page_num + 1,
                    image_bytes=image_bytes,
                    width=pix.width,
                    height=pix.height,
                    dpi=self.dpi,
                ))
                
                logger.debug(f"页面 {page_num + 1}/{total_pages} 转换完成")
            
            return page_images
            
        finally:
            doc.close()
    
    def convert_file(self, pdf_path: str) -> List[PageImage]:
        """
        将 PDF 文件转换为图片列表
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            页面图片列表
        """
        with open(pdf_path, "rb") as f:
            return self.convert_bytes(f.read())
    
    def get_page_image(self, pdf_bytes: bytes, page_num: int) -> Optional[PageImage]:
        """
        获取指定页面的图片
        
        Args:
            pdf_bytes: PDF 字节
            page_num: 页码（从1开始）
            
        Returns:
            页面图片或 None
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF 未安装")
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        try:
            if page_num < 1 or page_num > len(doc):
                return None
            
            page = doc[page_num - 1]
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat)
            
            if self.image_format == "jpeg":
                image_bytes = pix.tobytes("jpeg")
            else:
                image_bytes = pix.tobytes("png")
            
            return PageImage(
                page_num=page_num,
                image_bytes=image_bytes,
                width=pix.width,
                height=pix.height,
                dpi=self.dpi,
            )
            
        finally:
            doc.close()


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> List[bytes]:
    """
    快捷函数：PDF 转图片列表
    
    Args:
        pdf_bytes: PDF 字节
        dpi: 渲染 DPI
        
    Returns:
        图片字节列表
    """
    converter = PDFToImages(dpi=dpi)
    page_images = converter.convert_bytes(pdf_bytes)
    return [pi.image_bytes for pi in page_images]


def check_pdf_has_text(pdf_bytes: bytes, sample_pages: int = 3) -> Tuple[bool, float]:
    """
    检查 PDF 是否包含文本层
    
    Args:
        pdf_bytes: PDF 字节
        sample_pages: 采样页数
        
    Returns:
        (是否有文本, 文本覆盖率)
    """
    if not PYMUPDF_AVAILABLE:
        return True, 1.0  # 假设有文本
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        total_pages = len(doc)
        pages_to_check = min(sample_pages, total_pages)
        
        text_pages = 0
        total_text_len = 0
        
        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text().strip()
            
            if len(text) > 50:  # 超过50字符认为有文本
                text_pages += 1
                total_text_len += len(text)
        
        has_text = text_pages > 0
        coverage = text_pages / pages_to_check if pages_to_check > 0 else 0
        
        logger.info(f"PDF 文本检测: {text_pages}/{pages_to_check} 页有文本, 覆盖率 {coverage:.0%}")
        
        return has_text, coverage
        
    finally:
        doc.close()
