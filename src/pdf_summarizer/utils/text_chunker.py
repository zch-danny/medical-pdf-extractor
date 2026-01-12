"""
文本分块工具

用于将长文本分割成适合模型处理的小块
针对 Qwen3-8B 的 32K 上下文限制进行优化
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger
import re


# ===========================================
# Token 估算常量 (针对 Qwen3-8B)
# ===========================================
# Qwen3 使用 Byte-Pair Encoding (BPE)
# 中文: 平均 1 字符 ≈ 1.5-2 tokens
# 英文: 平均 1 单词 ≈ 1.3 tokens, 1 字符 ≈ 0.25-0.3 tokens
CHINESE_CHAR_TOKEN_RATIO = 1.8  # 中文字符到 token 的比例
ENGLISH_CHAR_TOKEN_RATIO = 0.3  # 英文字符到 token 的比例 (按字符)
ENGLISH_WORD_TOKEN_RATIO = 1.3  # 英文单词到 token 的比例

# Qwen3-8B 最大上下文
QWEN3_MAX_CONTEXT = 32768
# 安全输入限制 (留给 prompt 和输出)
DEFAULT_MAX_INPUT_TOKENS = 25000


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量 (针对 Qwen3 优化)
    
    Args:
        text: 输入文本
        
    Returns:
        估算的 token 数量
    """
    if not text:
        return 0
    
    # 分离中文和非中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
    non_chinese_text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]', '', text)
    
    # 中文字符 token 估算
    chinese_tokens = len(chinese_chars) * CHINESE_CHAR_TOKEN_RATIO
    
    # 英文/数字 token 估算 (按单词)
    english_words = len(non_chinese_text.split())
    english_tokens = english_words * ENGLISH_WORD_TOKEN_RATIO
    
    # 特殊字符和标点
    special_chars = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
    special_tokens = special_chars * 0.5
    
    total = int(chinese_tokens + english_tokens + special_tokens)
    return max(total, 1)


def tokens_to_chars(tokens: int, is_chinese: bool = True) -> int:
    """
    将 token 数量转换为字符数量估算
    
    Args:
        tokens: token 数量
        is_chinese: 是否主要是中文文本
        
    Returns:
        估算的字符数量
    """
    if is_chinese:
        return int(tokens / CHINESE_CHAR_TOKEN_RATIO)
    else:
        return int(tokens / ENGLISH_CHAR_TOKEN_RATIO)


@dataclass
class TextChunk:
    """文本块"""
    index: int
    text: str
    start_pos: int
    end_pos: int
    estimated_tokens: int = 0  # 估算的 token 数


class TextChunker:
    """
    文本分块器
    
    支持两种分块模式:
    - 字符模式 (默认): 按字符数分块
    - Token 模式: 按 token 数分块 (针对 Qwen3-8B 优化)
    """
    
    def __init__(
        self,
        chunk_size: int = 8000,
        chunk_overlap: int = 500,
        separators: List[str] = None,
        max_tokens: Optional[int] = None,
        use_token_mode: bool = False,
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 每块最大字符数 (字符模式)
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符优先级列表
            max_tokens: 每块最大 token 数 (token 模式, 默认 12000 适配 Qwen3-8B)
            use_token_mode: 是否使用 token 模式分块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_token_mode = use_token_mode
        # Token 模式参数: 每块 12K tokens, 留足够空间给 prompt 和输出
        self.max_tokens = max_tokens or 12000
        self.token_overlap = 300  # token 重叠
        
        self.separators = separators or [
            "\n\n",  # 段落
            "\n",    # 换行
            "。",    # 中文句号
            ". ",    # 英文句号+空格 (避免小数点)
            "；",    # 中文分号
            "; ",    # 英文分号
            "，",    # 中文逗号
            ", ",    # 英文逗号
            " ",     # 空格
        ]
    
    def split(self, text: str) -> List[TextChunk]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            
        Returns:
            文本块列表
        """
        if not text:
            return [TextChunk(index=0, text="", start_pos=0, end_pos=0, estimated_tokens=0)]
        
        # 根据模式选择分块策略
        if self.use_token_mode:
            return self._split_by_tokens(text)
        else:
            return self._split_by_chars(text)
    
    def _split_by_chars(self, text: str) -> List[TextChunk]:
        """
        按字符数分块 (原始方法)
        """
        if len(text) <= self.chunk_size:
            tokens = estimate_tokens(text)
            return [TextChunk(index=0, text=text, start_pos=0, end_pos=len(text), estimated_tokens=tokens)]
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # 计算当前块的结束位置
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # 如果不是最后一块，尝试在分隔符处断开
            if end_pos < len(text):
                best_break = self._find_best_break(text, current_pos, end_pos)
                if best_break > current_pos:
                    end_pos = best_break
            
            # 提取文本块
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                tokens = estimate_tokens(chunk_text)
                chunks.append(TextChunk(
                    index=chunk_index,
                    text=chunk_text,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    estimated_tokens=tokens,
                ))
                chunk_index += 1
            
            # 移动到下一块（考虑重叠）
            current_pos = end_pos - self.chunk_overlap
            if current_pos <= chunks[-1].start_pos if chunks else 0:
                current_pos = end_pos  # 避免无限循环
        
        total_tokens = sum(c.estimated_tokens for c in chunks)
        logger.info(f"文本分块完成 [字符模式]: {len(text)} 字符, ~{total_tokens} tokens -> {len(chunks)} 块")
        return chunks
    
    def _split_by_tokens(self, text: str) -> List[TextChunk]:
        """
        按 token 数分块 (针对 Qwen3-8B 优化)
        
        确保每块不超过 max_tokens，适合有限上下文模型
        """
        total_tokens = estimate_tokens(text)
        
        # 如果总 token 数在限制内，直接返回
        if total_tokens <= self.max_tokens:
            return [TextChunk(index=0, text=text, start_pos=0, end_pos=len(text), estimated_tokens=total_tokens)]
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        # 计算每块的大致字符数 (基于平均 token/字符比)
        # 对于英文为主的医学文献，平均约 0.3-0.5 tokens/字符
        avg_ratio = total_tokens / len(text) if len(text) > 0 else 0.5
        estimated_chunk_chars = int(self.max_tokens / max(avg_ratio, 0.3))
        overlap_chars = int(self.token_overlap / max(avg_ratio, 0.3))
        
        while current_pos < len(text):
            # 计算当前块的结束位置
            end_pos = min(current_pos + estimated_chunk_chars, len(text))
            
            # 检查实际 token 数，如果超过限制则调整
            chunk_text = text[current_pos:end_pos]
            chunk_tokens = estimate_tokens(chunk_text)
            
            # 如果超过 token 限制，缩小范围
            while chunk_tokens > self.max_tokens and end_pos > current_pos + 500:
                end_pos = current_pos + int((end_pos - current_pos) * 0.8)
                chunk_text = text[current_pos:end_pos]
                chunk_tokens = estimate_tokens(chunk_text)
            
            # 如果不是最后一块，尝试在分隔符处断开
            if end_pos < len(text):
                best_break = self._find_best_break(text, current_pos, end_pos)
                if best_break > current_pos + 200:  # 确保至少有 200 字符
                    end_pos = best_break
            
            # 提取文本块
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text:
                tokens = estimate_tokens(chunk_text)
                chunks.append(TextChunk(
                    index=chunk_index,
                    text=chunk_text,
                    start_pos=current_pos,
                    end_pos=end_pos,
                    estimated_tokens=tokens,
                ))
                chunk_index += 1
            
            # 移动到下一块（考虑重叠）
            next_pos = end_pos - overlap_chars
            
            # 防止无限循环
            if next_pos <= current_pos:
                current_pos = end_pos
            else:
                current_pos = next_pos
        
        total_chunks_tokens = sum(c.estimated_tokens for c in chunks)
        logger.info(
            f"文本分块完成 [Token模式]: {len(text)} 字符, ~{total_tokens} tokens -> {len(chunks)} 块, "
            f"每块上限 {self.max_tokens} tokens"
        )
        return chunks
    
    def _find_best_break(self, text: str, start: int, end: int) -> int:
        """
        在指定范围内找到最佳断点
        
        Args:
            text: 原始文本
            start: 起始位置
            end: 结束位置
            
        Returns:
            最佳断点位置
        """
        # 在结尾附近的范围内寻找分隔符
        search_start = max(start, end - self.chunk_overlap)
        search_text = text[search_start:end]
        
        for sep in self.separators:
            # 从后往前找最后一个分隔符
            last_sep = search_text.rfind(sep)
            if last_sep != -1:
                return search_start + last_sep + len(sep)
        
        # 没找到分隔符，返回原结束位置
        return end
    
    def estimate_chunks(self, text_length: int) -> int:
        """
        估算文本会被分成多少块
        
        Args:
            text_length: 文本长度
            
        Returns:
            预估块数
        """
        if text_length <= self.chunk_size:
            return 1
        
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        return (text_length - self.chunk_overlap) // effective_chunk_size + 1
