#!/usr/bin/env python3
"""测试v7.6混合策略提取器"""
import json, time, requests, sys
from pathlib import Path
from production_extractor_v7 import extract_pdf

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

def gpt_eval(result):
    if not result.get('success'):
        return 0, "提取失败"
    prompt = f"""评估医学文献提取质量(1-10分):
类型: {result.get('doc_type')}
结果: {json.dumps(result.get('result', {}), ensure_ascii=False)[:3000]}

评分标准:
- 10分: 完整准确，结构规范
- 8-9分: 主要信息完整，小问题
- 6-7分: 缺少重要信息或有错误
- 4-5分: 信息不完整
- 1-3分: 基本无效

只返回: 分数|简评 (如: 8.5|信息完整,格式规范)"""
    try:
        r = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"},
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}, timeout=60)
        text = r.json()['choices'][0]['message']['content']
        parts = text.split('|')
        return float(parts[0].strip()), parts[1].strip() if len(parts) > 1 else ""
    except Exception as e:
        return 0, str(e)

def main():
    pdf_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/root/autodl-tmp/pdf_input/pdf_input_09-1125/')
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    pdfs = sorted(pdf_dir.glob('*.pdf'))[:limit]
    print(f"测试 {len(pdfs)} 个PDF (v7.6混合策略)\n")
    
    results = []
    for i, pdf in enumerate(pdfs, 1):
        result = extract_pdf(str(pdf))
        stats = result.get('stats', {})
        doc_size = stats.get('doc_size', '?')
        total_p = stats.get('total_pages', '?')
        sel_p = stats.get('selected_pages', '?')
        
        score, comment = gpt_eval(result)
        results.append({
            'pdf': pdf.name, 'type': result.get('doc_type', '?'),
            'doc_size': doc_size, 'pages': f"{total_p}→{sel_p}",
            'score': score, 'time': result.get('time', 0)
        })
        
        print(f"[{i}/{len(pdfs)}] {pdf.name[:30]:<30} | {result.get('doc_type', '?'):<10} | {doc_size:<6} | {total_p}→{sel_p}页 | {score}/10 | {result.get('time', 0):.0f}s")
    
    # 统计
    scores = [r['score'] for r in results if r['score'] > 0]
    times = [r['time'] for r in results]
    sizes = {'short': 0, 'medium': 0, 'long': 0}
    for r in results:
        if r['doc_size'] in sizes:
            sizes[r['doc_size']] += 1
    
    print(f"\n{'='*60}")
    print(f"文档大小分布: short={sizes['short']}, medium={sizes['medium']}, long={sizes['long']}")
    print(f"平均评分: {sum(scores)/len(scores):.2f}/10 (≥8分: {len([s for s in scores if s >= 8])}/{len(scores)})")
    print(f"平均耗时: {sum(times)/len(times):.0f}s")

if __name__ == "__main__":
    main()
