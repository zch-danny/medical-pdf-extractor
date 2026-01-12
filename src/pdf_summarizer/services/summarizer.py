"""
摘要服务

核心业务逻辑：PDF解析、文本分块、摘要生成

增强功能:
- 并发 chunk 处理
- 医学文献专用摘要
- 进度回调支持
"""

import asyncio
import time
import re
from typing import Optional, Literal, Callable, Awaitable
from dataclasses import dataclass
from loguru import logger

from pdf_summarizer.core.config import settings
from pdf_summarizer.core.constants import (
    SUMMARY_PROMPTS,
    CHUNK_SUMMARY_PROMPT,
    MERGE_SUMMARY_PROMPT,
    MEDICAL_SUMMARY_PROMPTS,
    MEDICAL_CHUNK_SUMMARY_PROMPT,
    MEDICAL_MERGE_SUMMARY_PROMPT,
    MIN_TEXT_LENGTH,
)

from pdf_summarizer.core.exceptions import InsufficientContentError
from pdf_summarizer.core.summary_styles import (
    SummaryStyle,
    get_style_config,
    get_style_prompt,
)
from pdf_summarizer.models.llm_client import LLMClient
from pdf_summarizer.utils.pdf_parser import PDFParser
from pdf_summarizer.utils.text_chunker import TextChunker, TextChunk
from pdf_summarizer.utils.cache import SummaryCache


# 摘要策略类型
SummaryStrategy = Literal["auto", "stuff", "map_reduce", "refine"]

# 进度回调类型
ProgressCallback = Callable[[int, str], Awaitable[None]]

# 并发限制（防止 LLM 服务过载）
MAX_CONCURRENT_CHUNKS = 5


@dataclass
class SummaryResult:
    """摘要结果"""
    summary: str
    word_count: int
    language: str
    original_length: int
    processing_time: float
    chunks_processed: int
    style: str = "detailed"  # 摘要风格
    strategy: str = "auto"   # 使用的策略
    is_medical: bool = False  # 是否为医学文献模式


class SummarizerService:
    """
    摘要服务
    
    支持：
    - 直接文本摘要
    - PDF 文件摘要
    - 长文本分块摘要
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        pdf_parser: Optional[PDFParser] = None,
        text_chunker: Optional[TextChunker] = None,
        use_cache: bool = True,
    ):
        """
        初始化摘要服务
        
        Args:
            llm_client: LLM 客户端
            pdf_parser: PDF 解析器
            text_chunker: 文本分块器
            use_cache: 是否使用缓存
        """
        self.llm_client = llm_client or LLMClient()
        self.pdf_parser = pdf_parser or PDFParser(max_pages=settings.max_pdf_pages)
        self.text_chunker = text_chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_tokens=settings.max_chunk_tokens,
            use_token_mode=settings.use_token_mode,
        )
        self.cache = SummaryCache.get_instance() if use_cache else None
        
        logger.info("SummarizerService 初始化完成")

    def _postprocess_medical_summary(self, summary: str, language: str) -> str:
        if not summary:
            return summary

        text = summary.replace("\r\n", "\n").replace("\r", "\n")

        placeholder = "未报告" if language == "zh" else "Not Reported"
        if language == "zh":
            section_defs = [
                ("background", "【研究背景】", r"【\s*研究背景\s*】"),
                ("population", "【研究人群 (P)】", r"【\s*研究人群\s*\(\s*P\s*\)\s*】"),
                ("intervention", "【干预措施 (I) / 对照 (C)】", r"【\s*干预措施\s*\(\s*I\s*\)\s*/\s*对照\s*\(\s*C\s*\)\s*】"),
                ("outcomes", "【结果指标 (O)】", r"【\s*结果指标\s*\(\s*O\s*\)\s*】"),
                ("safety", "【安全性】", r"【\s*安\s*全\s*性\s*】"),
                ("conclusion", "【结论与局限性】", r"【\s*结论与局限性\s*】"),
            ]
            cutoff_markers = [
                "根据您提供",
                "我将按照",
                "严格遵循",
                "请根据以上",
            ]
        else:
            section_defs = [
                ("background", "【Background】", r"【\s*Background\s*】"),
                ("population", "【Population (P)】", r"【\s*Population\s*\(\s*P\s*\)\s*】"),
                ("intervention", "【Intervention (I) / Comparison (C)】", r"【\s*Intervention\s*\(\s*I\s*\)\s*/\s*Comparison\s*\(\s*C\s*\)\s*】"),
                ("outcomes", "【Outcomes (O)】", r"【\s*Outcomes\s*\(\s*O\s*\)\s*】"),
                ("safety", "【Safety】", r"【\s*Safety\s*】"),
                ("conclusion", "【Conclusion & Limitations】", r"【\s*Conclusion\s*&\s*Limitations\s*】"),
            ]
            cutoff_markers = [
                "Based on the provided",
                "I will",
                "Please",
            ]

        matches: list[tuple[int, int, str]] = []
        for key, _, pat in section_defs:
            for m in re.finditer(r"(^|\n)[#\s]*" + pat, text, flags=re.MULTILINE):
                matches.append((m.start(), m.end(), key))

        if not matches:
            cleaned = text.strip()
            for marker in cutoff_markers:
                pos = cleaned.find(marker)
                if pos != -1:
                    cleaned = cleaned[:pos].strip()

            parts: list[str] = []
            for idx, (_, heading, _) in enumerate(section_defs):
                content = cleaned if idx == 0 and cleaned else placeholder
                parts.append(f"{heading}\n{content}")
            return "\n\n".join(parts).strip()

        matches.sort(key=lambda x: x[0])

        first_match_by_key: dict[str, tuple[int, int]] = {}
        for start, end, key in matches:
            if key not in first_match_by_key:
                first_match_by_key[key] = (start, end)

        # Build content by taking the text between a section header and the next ANY section header.
        content_by_key: dict[str, str] = {}
        for idx, (start, end, key) in enumerate(matches):
            if key in content_by_key:
                continue
            next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
            content = text[end:next_start].strip()

            # Truncate if the model adds other bracket headings / markdown headings inside the section.
            generic_delim = re.search(r"(^|\n)\s*【[^】]{1,40}】", content)
            if generic_delim:
                content = content[: generic_delim.start()].strip()

            md_delim = re.search(r"(^|\n)\s*#{1,6}\s+", content)
            if md_delim:
                content = content[: md_delim.start()].strip()

            for marker in cutoff_markers:
                pos = content.find(marker)
                if pos != -1:
                    content = content[:pos].strip()

            content_by_key[key] = content

        parts: list[str] = []
        for key, heading, _ in section_defs:
            content = content_by_key.get(key, "").strip()
            if not content:
                content = placeholder
            parts.append(f"{heading}\n{content}")

        return "\n\n".join(parts).strip()
    
    async def summarize_text(
        self,
        text: str,
        language: Literal["zh", "en"] = "zh",
        max_length: int = 500,
        strategy: SummaryStrategy = "auto",
        is_medical: bool = False,
        style: SummaryStyle = "detailed",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SummaryResult:
        """
        对文本进行摘要
        
        Args:
            text: 待摘要文本
            language: 输出语言
            max_length: 摘要最大长度
            strategy: 摘要策略 (auto/stuff/map_reduce/refine)
            is_medical: 是否为医学文献模式
            progress_callback: 进度回调函数
            
        Returns:
            摘要结果
        """
        start_time = time.time()
        original_length = len(text)
        
        # 验证文本长度
        if original_length < MIN_TEXT_LENGTH:
            raise ValueError(f"文本太短，至少需要 {MIN_TEXT_LENGTH} 个字符")
        
        style_config = get_style_config(style)
        actual_max_length = max_length
        if style_config and style_config.max_length:
            actual_max_length = style_config.max_length

        actual_is_medical = is_medical or (style == "medical")

        # 尝试从缓存获取
        if self.cache:
            cached = await self.cache.get_summary(
                text,
                language,
                actual_max_length,
                style=style,
                is_medical=actual_is_medical,
                strategy=strategy,
            )
            if cached:
                logger.info(f"命中缓存: {original_length} 字符")
                return SummaryResult(
                    summary=cached["summary"],
                    word_count=cached["word_count"],
                    language=cached["language"],
                    original_length=cached["original_length"],
                    processing_time=0.0,  # 缓存命中无处理时间
                    chunks_processed=cached.get("chunks_processed", 1),
                    style=cached.get("style", style),
                    strategy=cached.get("strategy", strategy),
                    is_medical=cached.get("is_medical", actual_is_medical),
                )
        
        # 报告进度
        async def report_progress(progress: int, message: str):
            if progress_callback:
                await progress_callback(progress, message)
        
        await report_progress(10, "正在分析文本...")
        
        # 判断是否需要分块
        chunks = self.text_chunker.split(text)
        
        await report_progress(20, f"文本分块完成: {len(chunks)} 块")
        
        if len(chunks) == 1:
            if strategy == "auto":
                actual_strategy = "stuff"
            else:
                actual_strategy = strategy
            await report_progress(30, "正在生成摘要...")
            summary = await self._generate_summary(
                text=text,
                language=language,
                max_length=actual_max_length,
                is_medical=actual_is_medical,
                style=style,
            )
            chunks_processed = 1
        else:
            # 多块根据策略选择
            if strategy == "auto":
                actual_strategy = "map_reduce"  # 默认使用 map_reduce
            elif strategy == "stuff":
                actual_strategy = "map_reduce"  # stuff 不适用多块，降级为 map_reduce
            else:
                actual_strategy = strategy
            
            if actual_strategy == "refine":
                # Refine 策略：迭代式优化
                await report_progress(30, "使用 Refine 策略生成摘要...")
                summary = await self._summarize_refine(
                    chunks=chunks,
                    language=language,
                    max_length=actual_max_length,
                    is_medical=actual_is_medical,
                    style=style,
                    progress_callback=progress_callback,
                )
            else:
                # Map-Reduce 策略：并发分块处理
                await report_progress(30, "使用 Map-Reduce 策略生成摘要...")
                summary = await self._summarize_chunks_concurrent(
                    chunks=chunks,
                    language=language,
                    max_length=actual_max_length,
                    is_medical=actual_is_medical,
                    style=style,
                    progress_callback=progress_callback,
                )
            chunks_processed = len(chunks)
        
        processing_time = time.time() - start_time

        if actual_is_medical or style == "medical":
            summary = self._postprocess_medical_summary(summary, language)
        
        await report_progress(100, "摘要生成完成")
        
        # 计算字数
        word_count = len(summary) if language == "zh" else len(summary.split())
        
        mode_str = "医学文献" if actual_is_medical else "普通"
        logger.info(
            f"摘要完成 [{mode_str}]: {original_length} 字符 -> {word_count} 字, "
            f"耗时 {processing_time:.2f}s, 分块数 {chunks_processed}, 策略 {actual_strategy}"
        )
        
        result = SummaryResult(
            summary=summary,
            word_count=word_count,
            language=language,
            original_length=original_length,
            processing_time=round(processing_time, 2),
            chunks_processed=chunks_processed,
            style=style,
            strategy=actual_strategy,
            is_medical=actual_is_medical,
        )
        
        # 保存到缓存
        if self.cache:
            await self.cache.set_summary(
                text=text,
                language=language,
                max_length=actual_max_length,
                style=style,
                is_medical=actual_is_medical,
                strategy=actual_strategy,
                summary_data={
                    "summary": result.summary,
                    "word_count": result.word_count,
                    "language": result.language,
                    "original_length": result.original_length,
                    "chunks_processed": result.chunks_processed,
                    "style": result.style,
                    "strategy": result.strategy,
                    "is_medical": result.is_medical,
                }
            )
        
        return result
    
    async def summarize_pdf(
        self,
        pdf_bytes: bytes,
        language: Literal["zh", "en"] = "zh",
        max_length: int = 500,
        style: SummaryStyle = "detailed",
        strategy: SummaryStrategy = "auto",
        is_medical: bool = False,
    ) -> SummaryResult:
        """
        对 PDF 文件进行摘要
        
        Args:
            pdf_bytes: PDF 文件字节
            language: 输出语言
            max_length: 摘要最大长度
            
        Returns:
            摘要结果
        """
        # 解析 PDF
        text, metadata = self.pdf_parser.extract_text_from_bytes(pdf_bytes)
        
        stripped = text.strip() if text else ''
        if not stripped or len(stripped) < MIN_TEXT_LENGTH:
            raise InsufficientContentError(
                message="PDF 内容为空或太短，无法生成摘要",
                min_chars=MIN_TEXT_LENGTH,
                actual_chars=len(stripped),
            )
        
        logger.info(
            f"PDF 解析完成: {metadata['processed_pages']} 页, "
            f"{metadata['char_count']} 字符"
        )
        
        # 生成摘要
        return await self.summarize_text(
            text=text,
            language=language,
            max_length=max_length,
            style=style,
            strategy=strategy,
            is_medical=is_medical,
        )
    
    async def _generate_summary(
        self,
        text: str,
        language: str,
        max_length: int,
        is_medical: bool = False,
        style: SummaryStyle = "detailed",
    ) -> str:
        """
        生成单段文本的摘要
        
        Args:
            text: 待摘要文本
            language: 输出语言
            max_length: 摘要最大长度
        """

        if style == "medical" or is_medical:
            prompt = get_style_prompt("medical", text=text, language=language)
        else:
            prompt = get_style_prompt(style, text=text, language=language)

        summary = await self.llm_client.generate(
            prompt=prompt,
            max_tokens=max_length * 2,
            temperature=0.3,
        )

        return summary.strip()
    
    async def _summarize_chunks_concurrent(
        self,
        chunks: list[TextChunk],
        language: str,
        max_length: int,
        is_medical: bool = False,
        style: SummaryStyle = "detailed",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """
        并发分块摘要然后合并
        
        Args:
            chunks: 文本块列表
            language: 输出语言
            max_length: 最终摘要最大长度
            is_medical: 是否为医学文献模式
            progress_callback: 进度回调
            
        Returns:
            合并后的摘要
        """
        total_chunks = len(chunks)
        chunk_max_length = max(100, max_length // total_chunks)
        
        # 选择 prompt 模板
        if style == "medical" or is_medical:
            chunk_prompts = MEDICAL_CHUNK_SUMMARY_PROMPT
            merge_prompts = MEDICAL_MERGE_SUMMARY_PROMPT
        else:
            chunk_prompts = CHUNK_SUMMARY_PROMPT
            merge_prompts = MERGE_SUMMARY_PROMPT
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)
        completed_count = 0
        
        async def process_chunk(chunk: TextChunk) -> tuple[int, str]:
            """ 处理单个 chunk"""
            nonlocal completed_count
            
            async with semaphore:
                prompt_template = chunk_prompts.get(language, chunk_prompts["zh"])
                prompt = prompt_template.format(
                    chunk_index=chunk.index + 1,
                    total_chunks=total_chunks,
                    text=chunk.text,
                )
                
                chunk_summary = await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=chunk_max_length * 2,
                    temperature=0.3,
                )
                
                completed_count += 1
                
                # 报告进度
                if progress_callback:
                    progress = 30 + int((completed_count / total_chunks) * 50)
                    await progress_callback(progress, f"已处理 {completed_count}/{total_chunks} 块")
                
                logger.debug(f"块 {chunk.index + 1}/{total_chunks} 摘要完成")
                return chunk.index, chunk_summary.strip()
        
        # 并发处理所有 chunks
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 检查异常并按顺序排列结果
        chunk_summaries = [None] * total_chunks
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"块处理失败: {result}")
                raise result
            idx, summary = result
            chunk_summaries[idx] = summary
        
        if progress_callback:
            await progress_callback(85, "正在合并摘要...")
        
        # 合并所有块摘要
        all_summaries = "\n\n".join([
            f"[部分 {i+1}] {s}" 
            for i, s in enumerate(chunk_summaries)
        ])
        
        merge_prompt_template = merge_prompts.get(language, merge_prompts["zh"])
        merge_prompt = merge_prompt_template.format(
            summaries=all_summaries,
            max_length=max_length,
        )
        
        final_summary = await self.llm_client.generate(
            prompt=merge_prompt,
            max_tokens=max_length * 2,
            temperature=0.3,
        )
        
        return final_summary.strip()
    
    async def _summarize_chunks(
        self,
        chunks: list[TextChunk],
        language: str,
        max_length: int,
        is_medical: bool = False,
    ) -> str:
        """
        顺序分块摘要（兼容旧接口）
        """
        return await self._summarize_chunks_concurrent(
            chunks=chunks,
            language=language,
            max_length=max_length,
            is_medical=is_medical,
        )
    
    async def _summarize_refine(
        self,
        chunks: list[TextChunk],
        language: str,
        max_length: int,
        is_medical: bool = False,
        style: SummaryStyle = "detailed",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """
        Refine 策略：迭代式优化摘要
        
        流程: chunk1 -> 初始摘要 -> chunk2 + 摘要 -> 优化 -> ...
        比 Map-Reduce 更慢，但摘要质量更高、连贯性更好
        
        Args:
            chunks: 文本块列表
            language: 输出语言
            max_length: 最终摘要最大长度
            
        Returns:
            优化后的摘要
        """
        total_chunks = len(chunks)
        
        # Refine prompt 模板（内嵌，方便后续删除）
        REFINE_PROMPT = {
            "zh": """你是专业的文档摘要专家。请根据新增内容优化现有摘要。

【已有摘要】
{existing_summary}

【新增内容】（第 {chunk_index}/{total_chunks} 部分）
{new_content}

【要求】
1. 整合新内容中的关键信息到摘要中
2. 保持摘要的连贯性和逻辑性
3. 去除冗余信息
4. 控制在 {max_length} 字以内

【优化后的摘要】""",
            "en": """You are a professional document summarizer. Refine the existing summary with new content.

【Existing Summary】
{existing_summary}

【New Content】(Part {chunk_index}/{total_chunks})
{new_content}

【Requirements】
1. Integrate key information from new content
2. Maintain coherence and logic
3. Remove redundancy
4. Keep under {max_length} words

【Refined Summary】"""
        }

        MEDICAL_REFINE_PROMPT = {
            "zh": """你是一名专业的医学文献分析专家。请基于新增内容对已有的医学文献结构化摘要进行增量更新。

【已有摘要】
{existing_summary}

【新增内容】（第 {chunk_index}/{total_chunks} 部分）
{new_content}

【硬性规则】
1. 只能基于【新增内容】与【已有摘要】中出现的信息整合：不要编造、不要补全未提供的数据/结论
2. 数字必须原样保留：样本量、百分比、p值、置信区间(CI)、RR/OR/HR 等（不要改写、不要四舍五入）
3. 药物/治疗名称、剂量、给药途径、疗程必须原样保留（不自行推断）
4. 若某字段原文未明确给出，请写“未报告”（不要猜测）
5. 输出必须严格保持 6 个标题分段，且顺序固定：
   【研究背景】/【研究人群 (P)】/【干预措施 (I) / 对照 (C)】/【结果指标 (O)】/【安全性】/【结论与局限性】
6. 只输出最终摘要正文：不要复述规则/提示词，不要输出“注/说明/总结”等额外段落，不要输出分隔线/Markdown（例如：---、###）
7. 总长度控制在 {max_length} 字以内

【更新后的医学文献摘要】""",
            "en": """You are a professional medical literature analyst. Update the existing structured medical summary incrementally using the new content.

【Existing Summary】
{existing_summary}

【New Content】(Part {chunk_index}/{total_chunks})
{new_content}

【Hard Rules】
1. Only integrate information supported by the Existing Summary and New Content. Do not fabricate or fill in missing data/conclusions.
2. Preserve all numbers verbatim: sample sizes, percentages, p-values, confidence intervals (CI), RR/OR/HR, etc. Do not rewrite or round.
3. Preserve drug/treatment names, dosages, routes, and durations verbatim. Do not infer.
4. If a field is not explicitly reported, write "Not Reported". Do not guess.
5. Keep the output strictly sectioned with the following fixed order:
   【Background】/【Population (P)】/【Intervention (I) / Comparison (C)】/【Outcomes (O)】/【Safety】/【Conclusion & Limitations】
6. Output only the final summary text: do not repeat instructions/prompts, do not add extra notes, do not add separators/Markdown (e.g., --- or ###)
7. Keep under {max_length} words

【Updated Medical Literature Summary】"""
        }
        
        # 初始摘要（第一块）
        current_summary = await self._generate_summary(
            text=chunks[0].text,
            language=language,
            max_length=max_length,
            is_medical=is_medical,
            style=style,
        )
        logger.debug(f"Refine: 初始摘要完成 (块 1/{total_chunks})")

        if progress_callback:
            await progress_callback(35, f"Refine: 已完成 1/{total_chunks} 块")
        
        # 迭代优化（后续块）
        for i, chunk in enumerate(chunks[1:], start=2):
            prompt_dict = MEDICAL_REFINE_PROMPT if (style == "medical" or is_medical) else REFINE_PROMPT
            prompt_template = prompt_dict.get(language, prompt_dict["zh"])
            prompt = prompt_template.format(
                existing_summary=current_summary,
                chunk_index=i,
                total_chunks=total_chunks,
                new_content=chunk.text,
                max_length=max_length,
            )
            
            current_summary = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_length * 2,
                temperature=0.3,
            )
            current_summary = current_summary.strip()

            if progress_callback:
                progress = 35 + int((i / total_chunks) * 45)
                await progress_callback(progress, f"Refine: 已完成 {i}/{total_chunks} 块")
            
            logger.debug(f"Refine: 块 {i}/{total_chunks} 优化完成")
        
        return current_summary
