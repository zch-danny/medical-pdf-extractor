#!/usr/bin/env python3
"""三阶段架构评估 - 与单阶段对比"""
import json, time, requests, fitz, re
from pathlib import Path
from datetime import datetime
from three_stage_extractor import three_stage_extract

GPT_API = "https://api.bltcy.ai/v1/chat/completions"
GPT_KEY = "sk-XmnO3SWrXmVdB4BLGnDIZ318PVQEliOcaUQ5eLvqIxQKLEn7"

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

def gpt_evaluate(original_text, extract_result):
    """GPT评估"""
    es = json.dumps(extract_result, ensure_ascii=False, indent=2) if extract_result else "null"
    prompt = f"""评估提取质量(1-10分)：准确性、完整性、结构、页码准确、无编造。

原文:
{original_text}

提取结果:
{es[:4000]}

输出JSON: {{"scores":{{"accuracy":分数,"completeness":分数,"structure":分数,"page_accuracy":分数,"no_hallucination":分数,"overall":平均}},"summary":"总结"}}"""
    
    try:
        r = requests.post(GPT_API, headers={"Authorization": f"Bearer {GPT_KEY}"}, 
            json={"model": "gpt-5.2", "messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}, timeout=90)
        d = r.json()
        if "error" in d: return None
        return parse(d['choices'][0]['message']['content'])
    except:
        return None

def main():
    pdf_dir = Path('/root/autodl-tmp/pdf_summarization_data/pdf_samples')
    pdfs = list(pdf_dir.glob('*.pdf'))[:5]  # 测5个
    
    print("="*60)
    print("三阶段架构评估测试")
    print("="*60)
    
    results = []
    for i, pdf in enumerate(pdfs):
        print(f"\n[{i+1}/{len(pdfs)}] {pdf.name}")
        
        # 三阶段提取
        extract_result = three_stage_extract(str(pdf))
        
        # GPT评估
        original_text = get_text(pdf)
        eval_result = gpt_evaluate(original_text, extract_result.get('final_result'))
        
        score = eval_result.get('scores', {}).get('overall', 0) if eval_result else 0
        print(f"  评分: {score}/10")
        print(f"  耗时: {extract_result['performance']['total_time']:.1f}s")
        print(f"  Tokens: {extract_result['performance']['total_tokens']}")
        
        results.append({
            "file": pdf.name,
            "score": score,
            "time": extract_result['performance']['total_time'],
            "tokens": extract_result['performance']['total_tokens'],
            "api_calls": extract_result['performance']['api_calls'],
            "eval": eval_result
        })
    
    # 汇总
    scores = [r['score'] for r in results if r['score']]
    times = [r['time'] for r in results]
    tokens = [r['tokens'] for r in results]
    
    print("\n" + "="*60)
    print("汇总")
    print("="*60)
    print(f"评分: {scores}")
    print(f"平均分: {sum(scores)/len(scores):.1f}/10" if scores else "无")
    print(f">=8分: {len([s for s in scores if s>=8])}/{len(scores)}")
    print(f"平均耗时: {sum(times)/len(times):.1f}s")
    print(f"平均tokens: {sum(tokens)//len(tokens)}")
    
    # 保存
    output_dir = Path('/root/extraction_test_results/three_stage_test')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"eval_{ts}.json", 'w') as f:
        json.dump({"results": results, "avg_score": sum(scores)/len(scores) if scores else 0}, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果: {output_dir}")

if __name__ == '__main__':
    main()
