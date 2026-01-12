# 综述/评论文献信息提取器 v2

## 任务
从综述、评论、解读类文献中提取结构化信息。

## 关键规则

### 页码标注
- `[第1页]` → `"p1"`，`[第2页]` → `"p2"`
- **必须核实每条信息实际出现在哪一页**
- 禁止默认填p1

### 完整性
- 提取摘要中的所有关键数据点
- 提取主要结论和临床意义
- 如某字段找不到，填 `"Not Found"`

## 输出格式

```json
{
  "doc_metadata": {
    "title": "",
    "authors": "",
    "journal": "",
    "year": "",
    "doi": "",
    "article_type": "Review/Commentary/Editorial/Letter",
    "sources": []
  },
  "abstract_summary": {
    "background": "",
    "objective": "",
    "main_findings": [],
    "conclusion": "",
    "sources": []
  },
  "key_data_points": [
    {
      "description": "数据描述",
      "value": "具体数值",
      "sources": []
    }
  ],
  "clinical_implications": {
    "main_message": "",
    "recommendations": [],
    "sources": []
  },
  "extraction_quality": {
    "completeness": "High/Medium/Low",
    "notes": ""
  }
}
```

只输出JSON。

【文献内容】
