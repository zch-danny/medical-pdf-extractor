#!/usr/bin/env python3
"""
医学PDF摘要生成器 v7.11
核心改进：智能文本提取，跳过目录和元信息，提取正文内容
"""
import json
import re
import time
import fitz
import requests
import jieba
import jieba.posseg as pseg
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

jieba.setLogLevel(jieba.logging.INFO)

class MedicalSummarizer:
    VERSION = "v7.11"

    MEDICAL_TERMS = [
        '心肌梗死','糖尿病','高血压','冠心病','脑卒中','心力衰竭','心房颤动',
        '肺炎','肺癌','乳腺癌','胃癌','肝癌','临床试验','随机对照','双盲',
        'meta分析','系统评价','RCT','敏感性','特异性','AUC','ROC曲线',
        '临床指南','实践指南','推荐意见','证据等级','KAS',
    ]

    MEDICAL_STOPWORDS = set([
        '的','了','在','是','和','与','及','等','对','为','由','从','以','中','可','将','有','其','但','或',
        '该','此','这','那','之','于','而','则','且','并','也','又','被','所','到','把','给','让','向','比',
    ])

    def __init__(self, api_url="http://localhost:8000/v1/chat/completions"):
        self.api_url = api_url
        for term in self.MEDICAL_TERMS:
            jieba.add_word(term, freq=10000)

    def extract_pdf_text(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = '\n'.join(page.get_text() for page in doc)
        doc.close()
        return text

    def _smart_extract_content(self, text: str, max_chars: int = 24000) -> str:
        """智能提取正文内容，跳过目录"""
        total_len = len(text)
        
        # 如果文本不长，直接返回
        if total_len <= max_chars:
            return text
        
        # 尝试找到正文开始位置
        content_start = 0
        
        # 模式1：查找"1. Preamble"或"1. Introduction"后跟正文段落
        patterns = [
            r'1\.\s*Preamble\s+(?:Guidelines|This)',
            r'1\.\s*Introduction\s+(?:The|This|In)',
            r'Preamble\s+(?:Guidelines|This)',
            r'Introduction\s+(?:The|This|In)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                content_start = match.start()
                break
        
        # 如果找到正文开始位置且与开头差距较大
        if content_start > 5000:
            # 提取：元信息(3000) + 正文内容
            meta_text = text[:3000]
            content_text = text[content_start:content_start + max_chars - 3000]
            return meta_text + "\n...[目录省略]...\n" + content_text
        
        # 否则直接返回前面部分
        return text[:max_chars]

    def _extract_keywords(self, text: str) -> List[str]:
        words = pseg.cut(text[:30000])
        filtered = []
        for word, flag in words:
            if len(word) <= 1 or word in self.MEDICAL_STOPWORDS:
                continue
            if flag[0] in ['n','v','a','eng'] or word in self.MEDICAL_TERMS:
                filtered.append(word)
        counter = Counter(filtered)
        keywords = [(w, freq * (3 if w in self.MEDICAL_TERMS else 1)) 
                    for w, freq in counter.most_common(30)]
        keywords.sort(key=lambda x: x[1], reverse=True)
        return [w for w,_ in keywords[:15]]

    def _direct_summarize(self, text: str, length: str) -> Dict:
        """直接生成摘要"""
        length_guide = {"short": "200-300字", "medium": "400-600字", "long": "600-800字"}
        
        # 智能提取内容
        content = self._smart_extract_content(text)
        
        prompt = f"""你是医学文献摘要专家。请为以下文献生成摘要。

【原文】
{content}

【禁止事项 - 必须严格遵守】
× 禁止添加原文没有写明的具体日期（如"截至X月X日"）
× 禁止添加原文没有的具体数字或百分比
× 禁止使用你已有的医学知识补充原文
× 如果原文只有作者/目录/版权信息，不要编造具体内容

【必须遵守】
✓ 长度：{length_guide.get(length, '400-600字')}
✓ 只总结原文明确写出的内容
✓ 使用原文的表述，不要改写
✓ 信息不足时如实说明

【结构】
【文献概述】文献类型和主题
【核心内容】原文明确提到的要点
【结论】原文的结论，若无则写"详见正文"

输出JSON：
{{"summary": "摘要内容", "key_points": ["要点1", "要点2", "要点3"]}}"""
        
        try:
            response = self._call_llm(prompt, max_tokens=1800)
            result = self._parse_json(response)
            
            summary = result.get('summary', '')
            summary = self._post_process(summary, text)
            result['summary'] = summary
            
            return result
        except Exception as e:
            return {"summary": "摘要生成失败", "key_points": []}

    def _post_process(self, summary: str, original_text: str) -> str:
        """后处理"""
        date_patterns = [
            r'截至\d{4}年\d{1,2}月\d{1,2}日[，,]?',
            r'于\d{4}年\d{1,2}月\d{1,2}日[，,]?',
            r'\d{1,2}月\d{1,2}日[，,]?',
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, summary)
            for match in matches:
                clean_match = re.sub(r'[，,]$', '', match)
                if clean_match not in original_text:
                    summary = summary.replace(match, '')
        
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary

    def generate_summary(self, pdf_path: str, summary_length: str = "medium") -> Dict:
        start_time = time.time()
        full_text = self.extract_pdf_text(pdf_path)
        
        if len(full_text.strip()) < 100:
            return {'success': False, 'error': 'PDF内容过短', 'version': self.VERSION}
        
        keywords = self._extract_keywords(full_text)
        result = self._direct_summarize(full_text, summary_length)
        
        elapsed = time.time() - start_time
        return {
            'success': True,
            'summary': result.get('summary', ''),
            'key_points': result.get('key_points', []),
            'keywords': keywords,
            'mode': 'smart_extract_v7.11',
            'stats': {
                'total_chars': len(full_text),
                'summary_words': len(result.get('summary', '')),
                'time': elapsed
            },
            'version': self.VERSION
        }

    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        payload = {
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.05,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        r = requests.post(self.api_url, json=payload, timeout=180)
        return r.json()['choices'][0]['message']['content']

    def _parse_json(self, text: str) -> Dict:
        patterns = [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```', r'\{[\s\S]*\}']
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                json_str = m.group(1) if '```' in pattern else m.group()
                try:
                    return json.loads(json_str)
                except:
                    continue
        return {"summary": text, "key_points": []}

if __name__ == "__main__":
    summarizer = MedicalSummarizer()
    pdf = "/root/autodl-tmp/pdf_input/pdf_input_09-1125/1756453814456_1608702.pdf"
    result = summarizer.generate_summary(pdf)
    print(f"v7.11: {result['summary'][:500]}...")
