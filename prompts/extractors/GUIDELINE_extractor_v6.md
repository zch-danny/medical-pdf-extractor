# 临床指南信息提取器 v6

## 任务
从医学指南/共识文献中提取结构化信息。

## 关键规则

### 页码标注（最重要）
- 每条信息必须标注**实际出现的页码**
- 输入文本中的 `[第1页]` `[第2页]` 对应 `"p1"` `"p2"`
- **先找到[第X页]标记，再确定该内容属于哪一页**
- 禁止默认填p1，必须逐条核实

### 完整性要求
- **必须提取recommendations**，即使原文没有明确标号
- 推荐性陈述包括：建议、推荐、should、recommend、考虑、要求
- 如果5页内没有推荐，明确写 `"recommendations": [{"id": "N/A", "text": "No recommendations found in provided pages", "sources": []}]`

### 准确性要求
- 只提取原文存在的信息
- 不确定时填 `"Not Found"`
- 禁止推测或补充

## 示例

### 输入片段
```
[第1页]
Technology appraisal guidance
Published: 3 September 2025
www.nice.org.uk/guidance/ta1096

[第2页]
Your responsibility
The recommendations in this guidance...

[第3页]
Contents
1 Recommendations ........................ 4

[第4页]
1 Recommendations
1.1 Benralizumab as an add-on to standard care can be used...
```

### 正确输出
```json
{
  "doc_metadata": {
    "title": "Technology appraisal guidance",
    "organization": "NICE",
    "publish_date": "3 September 2025",
    "version": "TA1096",
    "url": "www.nice.org.uk/guidance/ta1096",
    "sources": ["p1"]
  },
  "recommendations": [
    {
      "id": "1.1",
      "text": "Benralizumab as an add-on to standard care can be used...",
      "sources": ["p4"]
    }
  ]
}
```
注意：推荐1.1在第4页出现，所以sources是p4而不是p1。

## 输出格式

```json
{
  "doc_metadata": {
    "title": "",
    "authors": "",
    "organization": "",
    "publish_date": "",
    "year": "",
    "version": "",
    "doi": "",
    "document_type": "",
    "url": "",
    "sources": []
  },
  "scope": {
    "objective": "",
    "target_population": "",
    "setting": "",
    "sources": []
  },
  "recommendations": [
    {
      "id": "编号",
      "text": "原文内容，逐字复制",
      "strength": "仅当原文标注时填写",
      "sources": ["该推荐实际出现的页码"]
    }
  ],
  "key_findings": [
    {
      "finding": "",
      "sources": []
    }
  ],
  "extraction_quality": {
    "pages_covered": "p1-p5",
    "completeness": "High/Medium/Low",
    "notes": ""
  }
}
```

## 检查清单（输出前核对）
1. [ ] 每个sources都核实过对应的[第X页]标记？
2. [ ] recommendations不为空？
3. [ ] 没有编造原文不存在的信息？

只输出JSON，不要解释。

【文献内容】