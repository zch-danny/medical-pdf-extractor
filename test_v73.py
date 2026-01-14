#!/usr/bin/env python3
import json, requests, sys
from pathlib import Path
from production_extractor_v73 import extract_pdf

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

def gpt_eval(result):
    if not result.get('success'): return 0
    prompt = f"""评估医学文献提取质量(1-10分):
类型: {result.get('doc_type')}
结果: {json.dumps(result.get('result', {}), ensure_ascii=False)[:3000]}

评分:10=完整准确,8-9=主要完整,6-7=有缺失,4-5=不完整,1-3=无效
只返回数字"""
    try:
        r = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"},
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 20}, timeout=60)
        return float(r.json()['choices'][0]['message']['content'].strip().split()[0])
    except: return 0

def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/autodl-tmp/pdf_input/pdf_input_09-1125/')
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    pdfs = sorted(pdf_dir.glob('*.pdf'))[:limit]
    print(f"测试 {len(pdfs)} 个PDF (v7.3智能选择)\n")
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        result = extract_pdf(str(pdf))
        stats = result.get('stats', {})
        score = gpt_eval(result)
        results.append({'score': score, 'time': result.get('time', 0), 'pages': stats.get('total_pages', 0)})
        print(f"[{i}/{len(pdfs)}] {pdf.name[:30]:<30} | {result.get('doc_type', '?'):<10} | {stats.get('total_pages', '?')}→{stats.get('selected_pages', '?')}页 | {score}/10 | {result.get('time', 0):.0f}s")
    
    scores = [r['score'] for r in results if r['score'] > 0]
    short = [r['score'] for r in results if r['pages'] <= 15 and r['score'] > 0]
    medium = [r['score'] for r in results if 15 < r['pages'] <= 50 and r['score'] > 0]
    long_docs = [r['score'] for r in results if r['pages'] > 50 and r['score'] > 0]
    
    print(f"\n{'='*70}")
    print(f"总体: {sum(scores)/len(scores):.2f}/10 | ≥8分: {len([s for s in scores if s >= 8])}/{len(scores)} | 耗时: {sum([r['time'] for r in results])/len(results):.0f}s")
    if short: print(f"  short(≤15页): {sum(short)/len(short):.2f}/10 ({len(short)}个)")
    if medium: print(f"  medium(16-50页): {sum(medium)/len(medium):.2f}/10 ({len(medium)}个)")
    if long_docs: print(f"  long(>50页): {sum(long_docs)/len(long_docs):.2f}/10 ({len(long_docs)}个)")

if __name__ == "__main__":
    main()
