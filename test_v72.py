#!/usr/bin/env python3
"""测试v7.2版本（只读前15页）"""
import json, time, requests, sys, fitz, re
from pathlib import Path

LOCAL_API = "http://localhost:8000/v1/chat/completions"
GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"
FEWSHOT_DIR = Path(__file__).parent / "fewshot_samples"

def clean_json_string(s):
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def extract_first_json(text):
    text = re.sub(r'^```json\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    text = clean_json_string(text)
    start = text.find('{')
    if start == -1: return None
    depth, in_string, escape = 0, False, False
    for i, c in enumerate(text[start:], start):
        if escape: escape = False; continue
        if c == '\\': escape = True; continue
        if c == '"' and not escape: in_string = not in_string; continue
        if in_string: continue
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0: return text[start:i+1]
    return None

def call_llm(prompt, max_tokens=6000):
    resp = requests.post(LOCAL_API, json={
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=300)
    return resp.json()["choices"][0]["message"]["content"]

def extract_text_v72(pdf_path, max_pages=15):
    """v7.2方式：只读前15页"""
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages = min(max_pages, total)
    text = "\n".join([f"=== 第{i+1}页 ===\n{doc[i].get_text()}" for i in range(pages)])
    doc.close()
    return text, {'total_pages': total, 'selected_pages': pages, 'doc_size': 'v7.2-first15'}

def load_fewshot(doc_type):
    path = FEWSHOT_DIR / f"{doc_type}_sample.json"
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None

def classify(text):
    prompt = "判断医学文档类型，返回: GUIDELINE/REVIEW/OTHER\n- GUIDELINE: 临床指南/诊疗规范/技术评估\n- REVIEW: 综述/系统评价/Meta分析\n- OTHER: 其他\n\n文档开头：\n" + text[:2500] + "\n\n只返回类型名："
    result = call_llm(prompt, max_tokens=50)
    for t in ["GUIDELINE", "REVIEW", "OTHER"]:
        if t in result.upper(): return t
    return "OTHER"

def build_prompt(doc_type, text, fewshot):
    base = f"医学文献信息提取专家。从{doc_type}文档提取结构化信息。\n\n要求：\n1. sources标注页码如[\"p1\"]或[\"p3-p5\"]\n2. 只提取原文存在的信息，不编造\n3. 提取具体内容，不是章节标题\n4. 推荐/建议要完整引用原文\n\n"
    fewshot_section = ""
    if fewshot:
        ex = fewshot["expected_output"]
        simplified = {"doc_metadata": ex.get("doc_metadata", {})}
        if doc_type == "GUIDELINE" and "recommendations" in ex:
            simplified["recommendations"] = ex["recommendations"][:1]
        fewshot_section = f"示例：\n{json.dumps(simplified, ensure_ascii=False)[:1200]}\n\n"
    fmt_map = {
        "GUIDELINE": '{"doc_metadata":{...},"scope":{...},"recommendations":[{"id":"1.1","text":"完整推荐内容","strength":"强度","sources":["px"]}],"key_evidence":[...]}',
        "REVIEW": '{"doc_metadata":{...},"scope":{...},"key_findings":[{"id":"F1","finding":"具体发现","sources":["px"]}],"conclusions":[...]}',
        "OTHER": '{"doc_metadata":{...},"scope":{...},"key_findings":[...],conclusions":[...]}'
    }
    fmt = fmt_map.get(doc_type, fmt_map["OTHER"])
    if len(text) > 12000:
        text = text[:6000] + "\n\n...[中间部分省略]...\n\n" + text[-6000:]
    return f"{base}{fewshot_section}格式：{fmt}\n\n文档：\n{text}\n\n返回JSON："

def extract_v72(pdf_path):
    start = time.time()
    try:
        text, stats = extract_text_v72(pdf_path)
        if len(text.strip()) < 100:
            return {"success": False, "error": "内容过少", "time": time.time() - start}
        doc_type = classify(text)
        fewshot = load_fewshot(doc_type)
        prompt = build_prompt(doc_type, text, fewshot)
        result = call_llm(prompt)
        json_str = extract_first_json(result)
        if json_str:
            data = json.loads(json_str)
            return {"success": True, "doc_type": doc_type, "result": data, "time": time.time() - start, "stats": stats}
        return {"success": False, "error": "未找到JSON", "time": time.time() - start}
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start}

def gpt_eval(result):
    if not result.get('success'): return 0, "失败"
    prompt = f"""评估医学文献提取质量(1-10分):
类型: {result.get('doc_type')}
结果: {json.dumps(result.get('result', {}), ensure_ascii=False)[:3000]}

评分标准:
- 10分: 完整准确，结构规范
- 8-9分: 主要信息完整，小问题
- 6-7分: 缺少重要信息或有错误
- 4-5分: 信息不完整
- 1-3分: 基本无效

只返回: 分数|简评"""
    try:
        r = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"},
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}, timeout=60)
        text = r.json()['choices'][0]['message']['content']
        parts = text.split('|')
        return float(parts[0].strip()), parts[1].strip() if len(parts) > 1 else ""
    except: return 0, "评估失败"

def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/autodl-tmp/pdf_input/pdf_input_09-1125/')
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    pdfs = sorted(pdf_dir.glob('*.pdf'))[:limit]
    print(f"测试 {len(pdfs)} 个PDF (v7.2只读前15页)\n")
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        result = extract_v72(str(pdf))
        stats = result.get('stats', {})
        total_p = stats.get('total_pages', '?')
        sel_p = stats.get('selected_pages', '?')
        score, _ = gpt_eval(result)
        results.append({'score': score, 'time': result.get('time', 0), 'total': total_p})
        print(f"[{i}/{len(pdfs)}] {pdf.name[:30]:<30} | {result.get('doc_type', '?'):<10} | {total_p}→{sel_p}页 | {score}/10 | {result.get('time', 0):.0f}s")
    
    scores = [r['score'] for r in results if r['score'] > 0]
    times = [r['time'] for r in results]
    print(f"\n{'='*60}")
    print(f"平均评分: {sum(scores)/len(scores):.2f}/10 (≥8分: {len([s for s in scores if s >= 8])}/{len(scores)})")
    print(f"平均耗时: {sum(times)/len(times):.0f}s")

if __name__ == "__main__":
    main()
