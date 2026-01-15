#!/usr/bin/env python3
"""
改进版GPT评测脚本 - 基于提取结果引用的页码
用法: python evaluate_with_referenced_pages.py <pdf> [--api-key API_KEY] [--api-domain DOMAIN]

特点:
1. 只传给GPT提取结果引用到的页面原文，而非固定前10页
2. 对比v7.9和v7.11两个版本
"""

import os, sys, json, re, time
import fitz
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from production_extractor_v79 import MedicalPDFExtractor as V79
from production_extractor_v711_quote import MedicalPDFExtractorV711 as V711

PAGE_RE = re.compile(r'p(\d+)(?:\s*-\s*p?(\d+))?', re.I)

def parse_referenced_pages(result_dict, total_pages):
    """从提取结果中解析引用的页码"""
    pages = set()
    data = result_dict or {}
    
    for field in ['recommendations', 'key_findings', 'key_evidence']:
        items = data.get(field, []) or []
        for item in items:
            if not isinstance(item, dict):
                continue
            for src in (item.get('sources') or []):
                for m in PAGE_RE.finditer(str(src)):
                    a, b = m.group(1), m.group(2)
                    if a and not b:
                        pages.add(int(a))
                    elif a and b:
                        for x in range(int(a), int(b)+1):
                            pages.add(x)
    
    # 始终包含第1-3页(元数据) + 最后2页(结论)
    pages.update([1, 2, 3])
    if total_pages > 3:
        pages.update([total_pages-1, total_pages])
    
    return sorted(p for p in pages if 1 <= p <= total_pages)

def extract_pages_text(pdf_path, page_numbers, max_chars=12000):
    """提取指定页码的文本"""
    doc = fitz.open(pdf_path)
    texts = []
    for p in page_numbers:
        if 1 <= p <= len(doc):
            txt = doc[p-1].get_text()[:2000]  # 每页最多2000字
            texts.append(f"=== 第{p}页 ===\n{txt}")
    doc.close()
    combined = "\n".join(texts)
    return combined[:max_chars], page_numbers

def call_gpt(prompt, api_key, api_domain, max_tokens=500):
    """调用GPT API"""
    url = f"https://{api_domain}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-5.2",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    result = resp.json()
    if "error" in result:
        raise Exception(f"API错误: {result}")
    return result["choices"][0]["message"]["content"]

def evaluate_one(pdf_text, result_dict, ref_pages, api_key, api_domain):
    """用GPT评估单个提取结果"""
    prompt = f'''你是医学文献信息提取质量评估专家。

## 原文(提取结果引用的页面: {ref_pages}):
{pdf_text}

## AI提取结果:
{json.dumps(result_dict, ensure_ascii=False, indent=2)[:4000]}

## 评分标准(1-10分):
1. accuracy(准确性): 提取内容是否在原文中确实存在，无编造
2. completeness(完整性): 关键信息是否提取完整
3. sources(溯源准确): 页码标注是否准确对应原文位置

返回JSON格式: {{"accuracy": 分数, "completeness": 分数, "sources": 分数, "overall": 三项平均分, "comments": "简短评语"}}
只返回JSON:'''
    
    resp = call_gpt(prompt, api_key, api_domain, 400)
    # 提取JSON
    match = re.search(r'\{[^}]+\}', resp.replace('\n', ' '))
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"overall": 0, "error": "解析失败"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="基于引用页码的GPT评测")
    parser.add_argument("pdfs", nargs="+", help="PDF文件路径")
    parser.add_argument("--api-key", default=os.environ.get("GPT_API_KEY", ""), help="GPT API密钥")
    parser.add_argument("--api-domain", default="api.bltcy.ai", help="API域名")
    args = parser.parse_args()
    
    if not args.api_key:
        print("错误: 请设置GPT_API_KEY环境变量或使用--api-key参数")
        sys.exit(1)
    
    results = []
    for i, pdf in enumerate(args.pdfs, 1):
        print(f"\n[{i}/{len(args.pdfs)}] {os.path.basename(pdf)}")
        
        # 提取
        r79 = V79().extract(pdf)
        r711 = V711().extract(pdf)
        
        doc = fitz.open(pdf)
        total = len(doc)
        doc.close()
        
        row = {"pdf": pdf, "total_pages": total}
        
        for name, r in [("v79", r79), ("v711", r711)]:
            if not r.get("success"):
                row[name] = {"success": False, "score": None}
                print(f"  {name}: 提取失败")
                continue
            
            # 解析引用页码
            ref_pages = parse_referenced_pages(r.get("result"), total)
            pdf_text, _ = extract_pages_text(pdf, ref_pages)
            
            print(f"  {name}: 引用页={ref_pages[:8]}{'...' if len(ref_pages)>8 else ''}")
            
            # GPT评分
            try:
                score = evaluate_one(pdf_text, r.get("result"), ref_pages, args.api_key, args.api_domain)
                row[name] = {"success": True, "ref_pages": ref_pages, "score": score}
                print(f"  {name}: overall={score.get('overall', 0)}/10")
            except Exception as e:
                row[name] = {"success": True, "ref_pages": ref_pages, "error": str(e)}
                print(f"  {name}: 评分失败 - {e}")
        
        results.append(row)
    
    # 汇总
    print("\n=== 汇总 ===")
    for name in ["v79", "v711"]:
        scores = [r[name].get("score", {}).get("overall", 0) for r in results 
                  if r.get(name, {}).get("success") and r[name].get("score", {}).get("overall")]
        if scores:
            print(f"{name}: 平均={sum(scores)/len(scores):.2f}/10 (n={len(scores)})")
        else:
            print(f"{name}: 无有效评分")
    
    # 保存
    out_path = os.path.join(ROOT, "gpt_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
