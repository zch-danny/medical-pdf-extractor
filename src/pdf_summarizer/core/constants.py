"""
常量定义模块

包含 Prompt 模板和其他常量
"""

# ===========================================
# 摘要 Prompt 模板
# ===========================================

SUMMARY_PROMPTS = {
    "zh": """你是一个专业的文档摘要专家。请对以下文档内容进行摘要。

【要求】
1. 提取文档的核心观点和关键信息
2. 保留重要的数据、结论和发现
3. 使用清晰、专业的中文表达
4. 摘要长度控制在 {max_length} 字左右
5. 保持客观，不添加个人观点

【文档内容】
{text}

【中文摘要】""",

    "en": """You are a professional document summarization expert. Please summarize the following document.

【Requirements】
1. Extract core arguments and key information
2. Preserve important data, conclusions, and findings
3. Use clear, professional English
4. Keep the summary around {max_length} words
5. Remain objective, do not add personal opinions

【Document Content】
{text}

【Summary】""",
}

# ===========================================
# 医学文献摘要 Prompt 模板
# ===========================================

MEDICAL_SUMMARY_PROMPTS = {
    "zh": """你是一名专业的医学文献分析专家。请对以下医学文献进行结构化摘要。

【摘要结构要求】
请按照以下 PICO 框架组织摘要：

1. 【研究背景】
   - 研究目的和临床意义
   - 研究假设

2. 【研究人群 (Population)】
   - 纳入/排除标准
   - 样本量 (n=?)
   - 患者基线特征

3. 【干预措施 (Intervention) / 对照 (Comparison)】
   - 干预组：药物/治疗名称、剂量、用法、疗程
   - 对照组：安慰剂/标准治疗/无干预

4. 【结果指标 (Outcomes)】
   - 主要终点指标及结果
   - 次要终点指标及结果
   - 必须保留：具体数值、p值、置信区间(CI)、风险比(RR/OR/HR)
   - 统计显著性标注

5. 【安全性】
   - 不良反应/不良事件
   - 严重不良事件发生率

6. 【临床意义与局限性】
   - 临床实践启示
   - 研究局限性
   - 证据质量等级 (如适用)

【重要提示】
- 只能基于原文信息总结：不要编造、不要补全未提供的数据/结论
- 数字必须原样保留：样本量、百分比、p值、置信区间(CI)、RR/OR/HR 等
- 保留原文中的药物通用名/商品名、剂量、给药途径、疗程（不自行推断）
- 若某字段原文未明确给出，请写“未报告”
- 不要进行主观评价，只客观陈述研究结果
- 摘要长度控制在 {max_length} 字左右

【文献内容】
{text}

【医学文献摘要】""",

    "en": """You are a professional medical literature analyst. Please provide a structured summary of the following medical literature.

【Summary Structure Requirements】
Organize the summary according to the PICO framework:

1. 【Background】
   - Study objectives and clinical significance
   - Research hypothesis

2. 【Population (P)】
   - Inclusion/exclusion criteria
   - Sample size (n=?)
   - Patient baseline characteristics

3. 【Intervention (I) / Comparison (C)】
   - Intervention group: Drug/treatment name, dosage, administration, duration
   - Control group: Placebo/standard care/no intervention

4. 【Outcomes (O)】
   - Primary endpoints and results
   - Secondary endpoints and results
   - MUST preserve: specific values, p-values, confidence intervals (CI), risk ratios (RR/OR/HR)
   - Statistical significance notation

5. 【Safety】
   - Adverse reactions/adverse events
   - Serious adverse event incidence

6. 【Clinical Implications & Limitations】
   - Clinical practice implications
   - Study limitations
   - Evidence quality level (if applicable)

【Important Notes】
- Only summarize information supported by the source text. Do not fabricate or fill in missing data/conclusions.
- Preserve all numbers verbatim: sample sizes, percentages, p-values, confidence intervals (CI), RR/OR/HR, etc.
- Retain drug/treatment names, dosages, routes, and durations verbatim (do not infer).
- If a field is not explicitly reported, write "Not Reported".
- Do not make subjective evaluations, only objectively report study results
- Keep summary around {max_length} words

【Literature Content】
{text}

【Medical Literature Summary】""",
}

# 医学文献分块摘要 Prompt
MEDICAL_CHUNK_SUMMARY_PROMPT = {
    "zh": """你是一名医学文献分析专家。请对以下医学文献片段进行摘要。

【特别注意】
- 只能基于片段原文信息总结：不要编造、不要补全未提供的数据/结论
- 必须保留所有数字数据并原样抄录：样本量、百分比、p值、置信区间(CI)、RR/OR/HR
- 必须保留药物名称、剂量、给药途径、用法、疗程（不自行推断）
- 必须保留不良反应及其发生率（不自行推断）
- 识别并标注研究设计类型 (RCT/队列/病例对照等)
- 若片段未提供某信息，请不要猜测
- 只输出片段摘要正文：不要复述要求/提示词，不要输出分隔线/Markdown（例如：---、###），不要输出额外说明

【文献片段 {chunk_index}/{total_chunks}】
{text}

【片段摘要】""",

    "en": """You are a medical literature analyst. Please summarize the following medical literature segment.

【Special Attention】
- Only summarize information supported by the segment. Do not fabricate or fill in missing data/conclusions.
- MUST preserve all numbers verbatim: sample sizes, percentages, p-values, confidence intervals (CI), RR/OR/HR
- MUST preserve drug/treatment names, dosages, routes, regimens, and durations verbatim (do not infer)
- MUST preserve adverse reactions and their incidence rates verbatim (do not infer)
- Identify and note the study design type (RCT/cohort/case-control, etc.)
- If not reported in the segment, do not guess
- Output only the segment summary text: do not repeat instructions/prompts, do not add separators/Markdown (e.g., --- or ###), do not add extra commentary

【Literature Segment {chunk_index}/{total_chunks}】
{text}

【Segment Summary】""",

}

# 医学文献合并摘要 Prompt
MEDICAL_MERGE_SUMMARY_PROMPT = {
    "zh": """以下是一篇医学文献各部分的摘要。请将这些摘要整合为一个结构化的完整摘要。

【整合要求】
1. 按 PICO 框架组织: 研究背景/人群/干预/结果/安全性/结论
2. 确保统计数据完整准确 (p值、CI、样本量)
3. 保留所有药物信息和不良反应
4. 去除重复内容，保持逻辑连贯
5. 总摘要长度控制在 {max_length} 字左右
6. 只能基于各部分摘要中出现的信息整合：不要编造、不要补全未提供的数据/结论
7. 若某字段未在各部分摘要中出现，请写“未报告”
8. 只输出最终摘要正文：不要复述要求/提示词，不要输出分隔线/Markdown（例如：---、###），不要输出额外说明

【各部分摘要】
{summaries}

【完整医学文献摘要】""",

    "en": """Below are summaries of different sections of a medical literature. Please integrate them into a complete structured summary.

【Integration Requirements】
1. Organize by PICO framework: Background/Population/Intervention/Outcomes/Safety/Conclusion
2. Ensure statistical data is complete and accurate (p-values, CIs, sample sizes)
3. Preserve all drug information and adverse reactions
4. Remove redundancy while maintaining logical coherence
5. Keep total summary around {max_length} words
6. Only integrate information supported by the section summaries. Do not fabricate or fill in missing data/conclusions.
7. If a field is not explicitly reported in the section summaries, write "Not Reported".
8. Output only the final summary text: do not repeat instructions/prompts, do not add separators/Markdown (e.g., --- or ###), do not add extra commentary

【Section Summaries】
{summaries}

【Complete Medical Literature Summary】""",
}

# 分块摘要 Prompt（用于长文档）
CHUNK_SUMMARY_PROMPT = {
    "zh": """请对以下文档片段进行摘要，提取关键信息：

【文档片段 {chunk_index}/{total_chunks}】
{text}

【片段摘要】""",

    "en": """Please summarize the following document chunk, extracting key information:

【Document Chunk {chunk_index}/{total_chunks}】
{text}

【Chunk Summary】""",
}

# 合并摘要 Prompt
MERGE_SUMMARY_PROMPT = {
    "zh": """以下是一篇长文档各部分的摘要。请将这些摘要整合为一个连贯、完整的总摘要。

【要求】
1. 整合所有关键信息，避免重复
2. 保持逻辑连贯性
3. 总摘要长度控制在 {max_length} 字左右

【各部分摘要】
{summaries}

【总摘要】""",

    "en": """Below are summaries of different sections of a long document. Please merge them into a coherent, complete summary.

【Requirements】
1. Integrate all key information, avoid repetition
2. Maintain logical coherence
3. Keep the total summary around {max_length} words

【Section Summaries】
{summaries}

【Final Summary】""",
}

# ===========================================
# 支持的语言
# ===========================================

SUPPORTED_LANGUAGES = ["zh", "en"]

# ===========================================
# 文件类型
# ===========================================

SUPPORTED_FILE_TYPES = [".pdf"]
MIME_TYPES = {
    ".pdf": "application/pdf",
}

# ===========================================
# 默认值
# ===========================================

DEFAULT_CHUNK_SIZE = 8000  # 字符
DEFAULT_CHUNK_OVERLAP = 500  # 字符
DEFAULT_MAX_SUMMARY_LENGTH = 500  # 字/词
MIN_TEXT_LENGTH = 100  # 最小文本长度
