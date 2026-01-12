"""pdf_summarizer.core.exceptions

Project-level exception types.

We use these to distinguish between different classes of errors at the API layer
(e.g. mapping insufficient content to HTTP 422).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InsufficientContentError(ValueError):
    """Raised when extracted/received content is too short to summarize."""

    message: str = "内容为空或太短，无法生成摘要"
    min_chars: Optional[int] = None
    actual_chars: Optional[int] = None

    def __str__(self) -> str:
        details: list[str] = []
        if self.min_chars is not None:
            details.append(f"至少需要 {self.min_chars} 个字符")
        if self.actual_chars is not None:
            details.append(f"当前 {self.actual_chars} 个字符")

        if details:
            return f"{self.message}（{'，'.join(details)}）"
        return self.message
