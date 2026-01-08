#!/usr/bin/env python3
"""批量测试提取器 - 配置化版本"""
import json, time, requests, fitz, shutil, re
from pathlib import Path
from datetime import datetime

# ========== 配置 ==========
QWEN_API = "http://localhost:8000/v1/chat/completions"
GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "YOUR_API_KEY"  # 替换为实际key

PDF_DIR = Path('/path/to/pdf_samples')  # 替换为实际路径
OUT_ROOT = Path('./extraction_test_results')
PROMPT_DIR = Path('./prompts')

EXTRACTORS = {
    'GUIDELINE': 'extractors/GUIDELINE_extractor_v6.md',
    'META': 'extractors/META_extractor_v2.md',
    'REVIEW': 'extractors/REVIEW_extractor_v2.md',
    'OTHER': 'extractors/OTHER_extractor_v4.md'
}

# ========== 工具函数 ==========
def get_text(path, pages=5, chars=8000):
    """从PDF提取文本"""
    doc = fitz.open(path)
    parts = [f"\n[第{i+1}页]\n" + doc[i].get_text() for i in range(min(pages, len(doc)))]
    doc.close()
    return ''.join(parts)[:chars]

def parse(text):
    """解析JSON响应"""
    text = re.sub(r'```json\s*|\s*```', '', text)
    m = re.search(r'\{[\s\S]+\}', text)
    try: return json.loads(m.group()) if m else None
    except: return None

def qwen(prompt, tokens=4000):
    """调用Qwen API"""
    r = requests.post(QWEN_API, json={
        "model": "qwen3-8b", 
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, 
        "chat_template_kwargs": {"enable_thinking": False}
    }, timeout=180)
    d = r.json()
    return d['choices'][0]['message']['content'], d.get('usage', {}).get('total_tokens', 0)

def gpt_eval(prompt):
    """调用GPT进行评估"""
    try:
        r = requests.post(GPT_API, 
            headers={"Authorization": f"Bearer {GPT_KEY}"}, 
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}, 
            timeout=90)
        d = r.json()
        if "error" in d: return None, 0
        return d['choices'][0]['message']['content'], d.get('usage', {}).get('total_tokens', 0)
    except: 
        return None, 0

def process(pdf, out_dir, i, n, classifier_prompt):
    """处理单个PDF：分类->提取->评估"""
    print(f"[{i}/{n}] {pdf.name}", end=" ")
    t0 = time.time()
    
    # 创建输出目录并复制PDF
    (out_dir / pdf.stem).mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, out_dir / pdf.stem / pdf.name)
    
    text = get_text(pdf)
    
    # 1. 分类
    cc, ct = qwen(classifier_prompt.replace("【待分类文献】", f"【待分类文献】\n{text}"), 2000)
    cj = parse(cc)
    dtype = cj.get('type', 'OTHER') if cj else 'OTHER'
    
    # 2. 提取
    extractor_path = PROMPT_DIR / EXTRACTORS.get(dtype, "extractors/OTHER_extractor_v4.md")
    ep = extractor_path.read_text()
    ec, et = qwen(ep.replace("【文献内容】", f"【文献内容】\n{text}"), 4000)
    ej = parse(ec)
    
    # 3. 评估
    es = json.dumps(ej, ensure_ascii=False, indent=2) if ej else "null"
    evp = f"""评估提取质量(1-10分)：准确性、完整性、结构、页码准确、无编造。

原文:
{text}

提取结果:
{es}

输出JSON: {{"scores":{{"accuracy":分数,"completeness":分数,"structure":分数,"page_accuracy":分数,"no_hallucination":分数,"overall":平均}},"summary":"总结"}}"""
    
    evc, evt = gpt_eval(evp)
    evj = parse(evc) if evc else None
    score = evj.get('scores', {}).get('overall', 0) if evj else 0
    
    elapsed = time.time() - t0
    print(f"→ {dtype} | 评分:{score}/10 | {elapsed:.0f}s")
    
    # 保存结果
    result = {
        "file": pdf.name, 
        "type": dtype, 
        "score": score, 
        "time": elapsed,
        "extract": ej, 
        "eval": evj, 
        "tokens": {"qwen": ct+et, "gpt": evt}
    }
    
    with open(out_dir / pdf.stem / "result.json", 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"test_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    
    # 加载分类器提示词
    classifier_prompt = (PROMPT_DIR / 'classifier/classifier_prompt.md').read_text()
    
    pdfs = list(PDF_DIR.glob('*.pdf'))[:10]
    print(f"测试{len(pdfs)}个PDF → {out}\n")
    
    results = [process(p, out, i+1, len(pdfs), classifier_prompt) for i, p in enumerate(pdfs)]
    
    # 汇总
    scores = [r['score'] for r in results if r['score']]
    types = {}
    for r in results: 
        types[r['type']] = types.get(r['type'], 0) + 1
    
    print(f"\n{'='*50}")
    print(f"类型分布: {types}")
    print(f"评分: {scores}")
    print(f"平均分: {sum(scores)/len(scores):.1f}/10" if scores else "无有效评分")
    print(f">=8分: {len([s for s in scores if s>=8])}/{len(scores)}")
    
    # 保存汇总
    with open(out / "summary.json", 'w') as f:
        json.dump({
            "timestamp": ts,
            "scores": scores, 
            "avg": sum(scores)/len(scores) if scores else 0, 
            "types": types,
            "files": [{"f": r['file'], "t": r['type'], "s": r['score']} for r in results]
        }, f, indent=2)
    
    print(f"\n结果保存: {out}")

if __name__ == '__main__':
    main()