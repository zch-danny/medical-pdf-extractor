#!/usr/bin/env python3
"""Debug GPT evaluation"""

import sys
sys.path.insert(0, '/root/pdf_summarization_deploy_20251225_093847')

from stable_evaluator import evaluate_stable, call_gpt

# 简单测试GPT
test_prompt = "请返回JSON: {\"test\": 1}"
print("Testing GPT call...")
try:
    resp = call_gpt(test_prompt, max_tokens=50)
    print(f"GPT response: {resp}")
except Exception as e:
    print(f"GPT error: {e}")
