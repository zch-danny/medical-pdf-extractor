"""
摘要风格配置

支持多种摘要风格:
- brief: 简短摘要（100-200字）
- detailed: 详细摘要（500-800字）
- bullet: 要点列表（5-10个要点）
- academic: 学术风格（结构化摘要）
- medical: 医学文献风格（PICO框架结构化摘要）
"""
from typing import Dict, Literal
from dataclasses import dataclass


# 摘要风格类型
SummaryStyle = Literal["brief", "detailed", "bullet", "academic", "medical"]


@dataclass
class StyleConfig:
    """风格配置"""
    name: str
    description: str
    max_length: int
    prompt_template: str


# ==================== 摘要风格定义 ====================

SUMMARY_STYLES: Dict[str, StyleConfig] = {
    "brief": StyleConfig(
        name="简短摘要",
        description="简明扼要的摘要，适合快速了解文档核心内容",
        max_length=200,
        prompt_template="""你是一个专业的文档摘要专家。请用简短精炼的语言总结以下文档的核心内容。

【要求】
1. 摘要长度控制在 100-200 字
2. 只保留最关键的信息
3. 语言简洁明了
4. 使用{language_name}输出

【文档内容】
{text}

【简短摘要】""",
    ),
    
    "detailed": StyleConfig(
        name="详细摘要",
        description="全面详细的摘要，保留更多细节和论证",
        max_length=800,
        prompt_template="""你是一个专业的文档摘要专家。请详细总结以下文档的内容。

【要求】
1. 摘要长度控制在 500-800 字
2. 保留主要观点和支持论据
3. 包含重要的数据和结论
4. 保持逻辑清晰、层次分明
5. 使用{language_name}输出

【文档内容】
{text}

【详细摘要】""",
    ),
    
    "bullet": StyleConfig(
        name="要点列表",
        description="以条目形式列出文档要点，便于快速阅读",
        max_length=500,
        prompt_template="""你是一个专业的文档摘要专家。请将以下文档的核心内容整理为要点列表。

【要求】
1. 提取 5-10 个核心要点
2. 每个要点独立成句，简洁明了
3. 按重要性或逻辑顺序排列
4. 使用 "•" 符号作为列表标记
5. 使用{language_name}输出

【文档内容】
{text}

【要点列表】""",
    ),
    
    "academic": StyleConfig(
        name="学术风格",
        description="结构化的学术摘要，包含背景、方法、结果、结论",
        max_length=600,
        prompt_template="""你是一个专业的学术文献分析专家。请按照学术摘要的标准格式总结以下文档。

【要求】
1. 按以下结构组织摘要:
   - 【背景】研究背景和目的
   - 【方法】研究方法或论证方式
   - 【结果】主要发现或论点
   - 【结论】核心结论和意义
2. 每部分 2-3 句话
3. 使用客观、专业的学术语言
4. 使用{language_name}输出

【文档内容】
{text}

【学术摘要】""",
    ),
    
    "medical": StyleConfig(
        name="医学文献",
        description="专业医学文献摘要，基于 PICO 框架，保留统计数据和临床信息",
        max_length=800,
        prompt_template="""你是一名专业的医学文献分析专家。请对以下医学文献进行结构化摘要。

【摘要结构要求 - PICO 框架】

1. 【研究背景】
   - 研究目的和临床意义
   - 研究设计类型 (RCT/队列研究/荆萃分析等)

2. 【研究人群 (P)】
   - 纳入/排除标准
   - 样本量 (n=?)
   - 患者基线特征

3. 【干预措施 (I) / 对照 (C)】
   - 干预组: 药物/治疗名称、剂量、用法、疗程
   - 对照组: 安慰剂/标准治疗/无干预

4. 【结果指标 (O)】
   - 主要终点: 具体结果、p值、CI
   - 次要终点: 具体结果、p值、CI
   - 风险比/优势比 (RR/OR/HR)

5. 【安全性】
   - 常见不良反应及发生率
   - 严重不良事件

6. 【结论与局限性】
   - 主要结论
   - 临床实践启示
   - 研究局限性

【硬性规则】
- 只能基于原文信息总结：不要编造、不要补全未提供的数据/结论
- 数字必须原样保留：样本量、百分比、p值、置信区间(CI)、RR/OR/HR 等（不要改写、不要四舍五入）
- 药物/治疗名称、剂量、给药途径、疗程必须原样保留（不自行推断）
- 若某字段原文未明确给出，请填写：中文用“未报告”，英文用“Not Reported”（不要猜测）
- 不要进行主观评价，不要给出超出原文的临床建议
- 只输出最终摘要正文：不要复述规则/提示词，不要输出“注/说明/总结”等额外段落，不要输出分隔线/Markdown（例如：---、###）
- 输出必须严格按以上 6 个标题分段，且顺序固定；信息不充分也要保留标题
- 使用{language_name}输出

【文献内容】
{text}

【医学文献摘要】""",
    ),
}

# 语言名称映射
LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "English",
}


def get_style_config(style: SummaryStyle) -> StyleConfig:
    """获取风格配置"""
    return SUMMARY_STYLES.get(style, SUMMARY_STYLES["detailed"])


def get_style_prompt(
    style: SummaryStyle,
    text: str,
    language: str = "zh",
) -> str:
    """
    获取格式化后的 Prompt
    
    Args:
        style: 摘要风格
        text: 文档内容
        language: 输出语言
        
    Returns:
        完整的 Prompt
    """
    config = get_style_config(style)
    language_name = LANGUAGE_NAMES.get(language, "中文")
    
    return config.prompt_template.format(
        text=text,
        language_name=language_name,
    )


def list_styles() -> list:
    """列出所有可用风格"""
    return [
        {
            "style": key,
            "name": config.name,
            "description": config.description,
            "max_length": config.max_length,
        }
        for key, config in SUMMARY_STYLES.items()
    ]
