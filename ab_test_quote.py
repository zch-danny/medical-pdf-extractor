"""
A/B测试脚本: v7.9 (原版) vs v7.10 (Quote-then-Structure)
用法: python ab_test_quote.py <pdf_path> [pdf_path2] ...
      python ab_test_quote.py --dir <pdf_directory>
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# 导入两个版本的提取器
from production_extractor_v79 import MedicalPDFExtractor as ExtractorV79
from production_extractor_v710_quote import MedicalPDFExtractorV710 as ExtractorV710


def count_items(result: Dict) -> int:
    """统计提取的条目数"""
    if not result.get('success') or not result.get('result'):
        return 0
    data = result['result']
    count = 0
    for key in ['recommendations', 'key_findings', 'key_evidence']:
        if key in data and isinstance(data[key], list):
            count += len(data[key])
    return count


def has_sources(result: Dict) -> float:
    """计算有来源标注的条目比例"""
    if not result.get('success') or not result.get('result'):
        return 0.0
    data = result['result']
    total, with_source = 0, 0
    for key in ['recommendations', 'key_findings', 'key_evidence']:
        if key in data and isinstance(data[key], list):
            for item in data[key]:
                total += 1
                sources = item.get('sources', [])
                if sources and len(sources) > 0 and sources[0]:
                    with_source += 1
    return with_source / total if total > 0 else 0.0


def run_ab_test(pdf_path: str) -> Dict:
    """对单个PDF进行A/B测试"""
    print(f"\n{'='*60}")
    print(f"测试文件: {pdf_path}")
    print('='*60)
    
    results = {}
    
    # 测试 v7.9 (原版)
    print("\n[A] v7.9 原版提取中...")
    extractor_v79 = ExtractorV79()
    start = time.time()
    result_v79 = extractor_v79.extract(pdf_path)
    result_v79['time'] = time.time() - start
    results['v79'] = result_v79
    
    if result_v79['success']:
        print(f"    ✓ 成功 | 耗时: {result_v79['time']:.1f}s | 类型: {result_v79.get('doc_type')}")
        print(f"    条目数: {count_items(result_v79)} | 来源标注率: {has_sources(result_v79):.0%}")
    else:
        print(f"    ✗ 失败: {result_v79.get('error')}")
    
    # 测试 v7.10 (Quote-then-Structure)
    print("\n[B] v7.10 Quote-then-Structure 提取中...")
    extractor_v710 = ExtractorV710()
    start = time.time()
    result_v710 = extractor_v710.extract(pdf_path)
    result_v710['time'] = time.time() - start
    results['v710_quote'] = result_v710
    
    if result_v710['success']:
        print(f"    ✓ 成功 | 耗时: {result_v710['time']:.1f}s | 类型: {result_v710.get('doc_type')}")
        print(f"    条目数: {count_items(result_v710)} | 来源标注率: {has_sources(result_v710):.0%}")
    else:
        print(f"    ✗ 失败: {result_v710.get('error')}")
    
    # 比较
    print("\n--- 对比结果 ---")
    v79_items = count_items(result_v79)
    v710_items = count_items(result_v710)
    v79_sources = has_sources(result_v79)
    v710_sources = has_sources(result_v710)
    
    print(f"条目数:     v7.9={v79_items} vs v7.10={v710_items} {'(+)' if v710_items > v79_items else '(-)' if v710_items < v79_items else '(=)'}")
    print(f"来源标注率: v7.9={v79_sources:.0%} vs v7.10={v710_sources:.0%} {'(+)' if v710_sources > v79_sources else '(-)' if v710_sources < v79_sources else '(=)'}")
    print(f"耗时:       v7.9={result_v79.get('time',0):.1f}s vs v7.10={result_v710.get('time',0):.1f}s")
    
    return {
        'pdf': pdf_path,
        'v79': {
            'success': result_v79['success'],
            'time': result_v79.get('time', 0),
            'items': v79_items,
            'source_rate': v79_sources
        },
        'v710_quote': {
            'success': result_v710['success'],
            'time': result_v710.get('time', 0),
            'items': v710_items,
            'source_rate': v710_sources
        }
    }


def run_batch_test(pdf_paths: List[str]) -> None:
    """批量A/B测试"""
    all_results = []
    
    for pdf_path in pdf_paths:
        if not Path(pdf_path).exists():
            print(f"跳过不存在的文件: {pdf_path}")
            continue
        result = run_ab_test(pdf_path)
        all_results.append(result)
    
    if not all_results:
        print("没有可测试的文件")
        return
    
    # 汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    
    v79_success = sum(1 for r in all_results if r['v79']['success'])
    v710_success = sum(1 for r in all_results if r['v710_quote']['success'])
    
    v79_avg_time = sum(r['v79']['time'] for r in all_results) / len(all_results)
    v710_avg_time = sum(r['v710_quote']['time'] for r in all_results) / len(all_results)
    
    v79_avg_items = sum(r['v79']['items'] for r in all_results) / len(all_results)
    v710_avg_items = sum(r['v710_quote']['items'] for r in all_results) / len(all_results)
    
    v79_avg_sources = sum(r['v79']['source_rate'] for r in all_results) / len(all_results)
    v710_avg_sources = sum(r['v710_quote']['source_rate'] for r in all_results) / len(all_results)
    
    print(f"\n测试文件数: {len(all_results)}")
    print(f"\n成功率:")
    print(f"  v7.9:  {v79_success}/{len(all_results)} ({v79_success/len(all_results):.0%})")
    print(f"  v7.10: {v710_success}/{len(all_results)} ({v710_success/len(all_results):.0%})")
    
    print(f"\n平均耗时:")
    print(f"  v7.9:  {v79_avg_time:.1f}s")
    print(f"  v7.10: {v710_avg_time:.1f}s")
    
    print(f"\n平均条目数:")
    print(f"  v7.9:  {v79_avg_items:.1f}")
    print(f"  v7.10: {v710_avg_items:.1f}")
    
    print(f"\n平均来源标注率:")
    print(f"  v7.9:  {v79_avg_sources:.0%}")
    print(f"  v7.10: {v710_avg_sources:.0%}")
    
    # 结论
    print(f"\n{'='*60}")
    print("结论:")
    if v710_avg_sources > v79_avg_sources + 0.05:
        print("  ✓ v7.10 Quote-then-Structure 来源标注率显著提升")
    if v710_avg_items >= v79_avg_items * 0.9:
        print("  ✓ v7.10 条目提取数量保持稳定")
    if v710_avg_time <= v79_avg_time * 1.3:
        print("  ✓ v7.10 耗时在可接受范围内")
    print("="*60)
    
    # 保存详细结果
    output_file = "ab_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python ab_test_quote.py <pdf_path> [pdf_path2] ...")
        print("  python ab_test_quote.py --dir <pdf_directory>")
        sys.exit(1)
    
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("请指定PDF目录")
            sys.exit(1)
        pdf_dir = Path(sys.argv[2])
        pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]
        if not pdf_paths:
            print(f"目录 {pdf_dir} 中没有找到PDF文件")
            sys.exit(1)
    else:
        pdf_paths = sys.argv[1:]
    
    run_batch_test(pdf_paths)
