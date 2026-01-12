"""
PDF 解析工具

使用 PyMuPDF (fitz) 提取 PDF 文本
"""

import io
from typing import Optional, List, Tuple
from pathlib import Path
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF 未安装，PDF 解析功能不可用")


class PDFParser:
    """PDF 文档解析器"""
    
    def __init__(self, max_pages: int = 200):
        """
        初始化 PDF 解析器
        
        Args:
            max_pages: 最大处理页数
        """
        self.max_pages = max_pages
        
        if not PYMUPDF_AVAILABLE:
            raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")
    
    def extract_text_from_file(self, file_path: str) -> Tuple[str, dict]:
        """
        从 PDF 文件提取文本
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            (文本内容, 元数据)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"不支持的文件类型: {path.suffix}")
        
        return self._extract_text(fitz.open(file_path))
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Tuple[str, dict]:
        """
        从 PDF 字节流提取文本
        
        Args:
            pdf_bytes: PDF 文件字节内容
            
        Returns:
            (文本内容, 元数据)
        """
        pdf_stream = io.BytesIO(pdf_bytes)
        return self._extract_text(fitz.open(stream=pdf_stream, filetype="pdf"))
    
    def _extract_text(self, doc: "fitz.Document") -> Tuple[str, dict]:
        """
        从 PyMuPDF 文档对象提取文本
        
        Args:
            doc: fitz.Document 对象
            
        Returns:
            (文本内容, 元数据)
        """
        try:
            total_pages = len(doc)
            pages_to_process = min(total_pages, self.max_pages)
            
            text_parts = []
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                text = page.get_text("text")
                
                if text.strip():
                    text_parts.append(f"[p{page_num + 1}]\n{text}")
            
            full_text = "\n\n".join(text_parts)
            
            # 提取元数据
            metadata = {
                "total_pages": total_pages,
                "processed_pages": pages_to_process,
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "char_count": len(full_text),
            }
            
            logger.info(
                f"PDF 解析完成: {pages_to_process}/{total_pages} 页, "
                f"{len(full_text)} 字符"
            )
            
            return full_text, metadata
            
        finally:
            doc.close()
    
    def get_page_count(self, pdf_bytes: bytes) -> int:
        """获取 PDF 页数"""
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
