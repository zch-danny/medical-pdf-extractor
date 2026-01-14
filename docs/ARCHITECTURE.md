# 系统架构

## 处理流程
```
PDF → PyMuPDF解析 → 文档分类 → 策略选择 → 页面选择 → LLM提取 → JSON输出
```

## 提取策略
| 页数 | 策略 |
|-----|------|
| ≤50 | Single-Pass |
| 51-80 | Single-Long |
| >80 | MapReduce |

## 核心模块
- 文档分类器: LLM判断GUIDELINE/REVIEW/OTHER
- 页面选择器: 评分+过滤选择关键页
- MapReduce: 分块提取后合并

## 扩展点
- 新增API: 实现`_call_llm()`方法
- 新增文档类型: 添加分类规则+Few-shot示例
