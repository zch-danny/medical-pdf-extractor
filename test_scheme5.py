#!/usr/bin/env python3
"""测试方案5字段级检索提取器"""
import json, time, requests, sys
from pathlib import Path
from field_retrieval_extractor import extract_pdf

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

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
    print(f"测试 {len(pdfs)} 个PDF (方案5-字段级检索)\n")
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        result = extract_pdf(str(pdf))
        stats = result.get('stats', {})
        total_p = stats.get('total_pages', '?')
        chunks = stats.get('chunks', '?')
        score, _ = gpt_eval(result)
        results.append({'score': score, 'time': result.get('time', 0), 'total': total_p, 'chunks': chunks})
        print(f"[{i}/{len(pdfs)}] {pdf.name[:30]:<30} | {result.get('doc_type', '?'):<10} | {total_p}页/{chunks}块 | {score}/10 | {result.get('time', 0):.0f}s")
    
    scores = [r['score'] for r in results if r['score'] > 0]
    times = [r['time'] for r in results]
    print(f"\n{'='*60}")
    print(f"平均评分: {sum(scores)/len(scores):.2f}/10 (≥8分: {len([s for s in scores if s >= 8])}/{len(scores)})")
    print(f"平均耗时: {sum(times)/len(times):.0f}s")

if __name__ == "__main__":
    main()
