# API参考文档

## extract_pdf()
```python
def extract_pdf(pdf_path: str) -> Dict[str, Any]
```
返回: `{success, doc_type, result, time, stats}`

## MedicalPDFExtractor
```python
extractor = MedicalPDFExtractor(api_url="http://localhost:8000/v1/chat/completions")
result = extractor.extract("document.pdf")
doc_type = extractor.classify(text)
```

## SiliconFlowClient
```python
client = SiliconFlowClient(api_key="sk-xxx", model="qwen2.5-7b")
response = client.complete("prompt", max_tokens=4096)
```

## 输出结构
- `doc_metadata`: 标题、作者、年份
- `recommendations` (GUIDELINE): 推荐列表
- `key_findings` (REVIEW/OTHER): 关键发现
- `conclusions`: 结论
