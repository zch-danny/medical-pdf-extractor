# 使用指南

## Python调用
```python
from production_extractor_v79 import extract_pdf

result = extract_pdf("medical_guideline.pdf")
if result["success"]:
    print(f"类型: {result['doc_type']}")
    print(f"结果: {result['result']}")
```

## 命令行
```bash
python production_extractor_v79.py document.pdf
```

## 硅基流动API
```python
from siliconflow_client import SiliconFlowExtractor
extractor = SiliconFlowExtractor(api_key="sk-xxx")
result = extractor.extract("document.pdf")
```

## 输出格式
```json
{
    "success": true,
    "doc_type": "GUIDELINE",
    "result": {
        "doc_metadata": {"title": "...", "authors": []},
        "recommendations": [{"id": "1", "text": "...", "sources": ["p5"]}]
    },
    "time": 45.2
}
```

## 批量处理
```python
from pathlib import Path
for pdf in Path("pdfs/").glob("*.pdf"):
    result = extract_pdf(str(pdf))
```
