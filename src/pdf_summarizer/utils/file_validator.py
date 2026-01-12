"""
文件安全校验模块

功能:
1. PDF 文件魔数验证
2. 文件大小检查
3. 恶意内容检测
4. 文件元数据提取
"""
import io
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger

# PDF 文件魔数
PDF_MAGIC_BYTES = b'%PDF-'

# 常见恶意 PDF 特征
SUSPICIOUS_PATTERNS = [
    b'/JavaScript',
    b'/JS',
    b'/Launch',
    b'/EmbeddedFile',
    b'/OpenAction',
    b'/AA',  # Additional Actions
    b'/RichMedia',
]


@dataclass
class FileValidationResult:
    """文件验证结果"""
    is_valid: bool
    file_type: str
    file_size: int
    error: Optional[str] = None
    warnings: list = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size / (1024 * 1024), 2),
            "error": self.error,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class FileValidator:
    """
    文件安全验证器
    
    支持:
    - PDF 文件格式验证
    - 恶意内容检测
    - 文件大小限制
    """
    
    def __init__(
        self,
        max_size_mb: int = 50,
        check_suspicious: bool = True,
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_size_mb = max_size_mb
        self.check_suspicious = check_suspicious
    
    def validate_pdf(self, file_bytes: bytes, filename: str = "") -> FileValidationResult:
        """
        验证 PDF 文件
        
        Args:
            file_bytes: 文件字节内容
            filename: 文件名（用于日志）
            
        Returns:
            验证结果
        """
        file_size = len(file_bytes)
        warnings = []
        metadata = {"filename": filename}
        
        # 1. 检查文件大小
        if file_size > self.max_size_bytes:
            return FileValidationResult(
                is_valid=False,
                file_type="unknown",
                file_size=file_size,
                error=f"文件大小超过限制 ({self.max_size_mb}MB)",
            )
        
        if file_size < 100:
            return FileValidationResult(
                is_valid=False,
                file_type="unknown",
                file_size=file_size,
                error="文件太小，不是有效的 PDF",
            )
        
        # 2. 检查 PDF 魔数
        if not file_bytes.startswith(PDF_MAGIC_BYTES):
            return FileValidationResult(
                is_valid=False,
                file_type="unknown",
                file_size=file_size,
                error="文件不是有效的 PDF 格式（魔数验证失败）",
            )
        
        # 3. 提取 PDF 版本
        try:
            first_line = file_bytes[:20].decode('latin-1')
            if first_line.startswith('%PDF-'):
                pdf_version = first_line[5:8]
                metadata["pdf_version"] = pdf_version
        except Exception:
            pass
        
        # 4. 检查可疑内容
        if self.check_suspicious:
            suspicious_found = self._check_suspicious_content(file_bytes)
            if suspicious_found:
                warnings.extend(suspicious_found)
                logger.warning(f"PDF 包含可疑内容: {filename}, 特征: {suspicious_found}")
        
        # 5. 检查是否加密
        if b'/Encrypt' in file_bytes:
            warnings.append("PDF 可能被加密，解析可能受限")
        
        # 6. 估算页数（简单方法）
        page_count = file_bytes.count(b'/Type /Page') or file_bytes.count(b'/Type/Page')
        metadata["estimated_pages"] = max(1, page_count)
        
        return FileValidationResult(
            is_valid=True,
            file_type="application/pdf",
            file_size=file_size,
            warnings=warnings,
            metadata=metadata,
        )
    
    def _check_suspicious_content(self, file_bytes: bytes) -> list:
        """检查可疑内容"""
        found = []
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in file_bytes:
                found.append(pattern.decode('latin-1'))
        return found
    
    def validate_file_extension(self, filename: str) -> bool:
        """验证文件扩展名"""
        if not filename:
            return False
        return filename.lower().endswith('.pdf')


# 快捷函数
def validate_pdf_file(
    file_bytes: bytes,
    filename: str = "",
    max_size_mb: int = 50,
) -> FileValidationResult:
    """
    快捷验证函数
    
    Args:
        file_bytes: 文件内容
        filename: 文件名
        max_size_mb: 最大文件大小
        
    Returns:
        验证结果
    """
    validator = FileValidator(max_size_mb=max_size_mb)
    return validator.validate_pdf(file_bytes, filename)
