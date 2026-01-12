#!/usr/bin/env python3
"""批量测试v6/v4提取器 - 10个PDF"""
import json, time, requests, fitz, shutil, re
from pathlib import Path
from datetime import datetime

QWEN_API = "http://localhost:8000/v1/chat/completions"
GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

PDF_DIR = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
OUT_ROOT = Path('/root/extraction_test_results')

EXTRACTORS = {
    'GUIDELINE': 'GUIDELINE_extractor_v6.md',  # 新版
    'META': 'META_extractor_v2.md',
    'REVIEW': 'REVIEW_extractor_v2.md',  # 新版
    'OTHER': 'OTHER_extractor_v4.md'
}

def get_text(path, pages=5, chars=8000):
    doc = fitz.open(path)
    parts = [f"\n[第{i+1}页]\n" + doc[i].get_text() for i in range(min(pages, len(doc)))]
    doc.close()
    return ''.join(parts)[:chars]

def parse(text):
    text = re.sub(r'```json\s*|\s*```', '', text)
    m = re.search(r'\{[\s\S]+\}', text)
    try: return json.loads(m.group()) if m else None
    except: return None

def qwen(prompt, tokens=4000):
    r = requests.post(QWEN_API, json={"model": "qwen3-8b", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
    d = r.json()
    return d['choices'][0]['message']['content'], d.get('usage', {}).get('total_tokens', 0)

def gpt(prompt):
    try:
        r = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"}, 
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}, timeout=90)
        d = r.json()
        if "error" in d: return None, 0
        return d['choices'][0]['message']['content'], d.get('usage', {}).get('total_tokens', 0)
    except: return None, 0

def process(pdf, out_dir, i, n):
    print(f"[{i}/{n}] {pdf.name}", end=" ")
    t0 = time.time()
    
    (out_dir / pdf.stem).mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, out_dir / pdf.stem / pdf.name)
    
    text = get_text(pdf)
    
    # 分类
    cp = Path('/root/提示词/分类器/classifier_prompt.md').read_text()
    cc, ct = qwen(cp.replace("【待分类文献】", f"【待分类文献】\n{text}"), 2000)
    cj = parse(cc)
    dtype = cj.get('type', 'OTHER') if cj else 'OTHER'
    
    # 提取
    ep = Path(f'/root/提示词/专项提取器/{EXTRACTORS.get(dtype, "OTHER_extractor_v4.md")}').read_text()
    ec, et = qwen(ep.replace("【文献内容】", f"【文献内容】\n{text}"), 4000)
    ej = parse(ec)
    
    # 评估
    es = json.dumps(ej, ensure_ascii=False, indent=2) if ej else "null"
    evp = f"""评估提取质量(1-10分)：准确性、完整性、结构、页码准确、无编造。

原文:
{text}

提取结果:
{es}

输出JSON: {{"scores":{{"accuracy":分数,"completeness":分数,"structure":分数,"page_accuracy":分数,"no_hallucination":分数,"overall":平均}},"summary":"总结"}}"""
    
    evc, evt = gpt(evp)
    evj = parse(evc) if evc else None
    score = evj.get('scores', {}).get('overall', 0) if evj else 0
    
    elapsed = time.time() - t0
    print(f"→ {dtype} | 评分:{score}/10 | {elapsed:.0f}s")
    
    result = {"file": pdf.name, "type": dtype, "score": score, "time": elapsed,
              "extract": ej, "eval": evj, "tokens": {"qwen": ct+et, "gpt": evt}}
    
    with open(out_dir / pdf.stem / "result.json", 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"v6_test_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    
    pdfs = list(PDF_DIR.glob('*.pdf'))[:10]
    print(f"测试{len(pdfs)}个PDF → {out}\n")
    
    results = [process(p, out, i+1, len(pdfs)) for i, p in enumerate(pdfs)]
    
    scores = [r['score'] for r in results if r['score']]
    types = {}
    for r in results: types[r['type']] = types.get(r['type'], 0) + 1
    
    print(f"\n{'='*50}")
    print(f"类型: {types}")
    print(f"评分: {scores}")
    print(f"平均: {sum(scores)/len(scores):.1f}/10" if scores else "无")
    print(f">=8分: {len([s for s in scores if s>=8])}/{len(scores)}")
    
    with open(out / "summary.json", 'w') as f:
        json.dump({"scores": scores, "avg": sum(scores)/len(scores) if scores else 0, 
                   "types": types, "files": [{"f": r['file'], "t": r['type'], "s": r['score']} for r in results]}, f, indent=2)

if __name__ == '__main__':
    main()
