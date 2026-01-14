"""
硅基流动(SiliconFlow) API 客户端
兼容 OpenAI API 格式，可作为 vLLM 本地服务的替代方案

使用方法:
1. 设置环境变量: export SILICONFLOW_API_KEY=sk-xxxxxx
2. 直接调用: from siliconflow_client import SiliconFlowExtractor
"""
import os
import json
import re
import time
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"
SUPPORTED_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "glm4-9b": "THUDM/glm-4-9b-chat",
}
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class SiliconFlowClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, base_url: str = SILICONFLOW_API_BASE):
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 SILICONFLOW_API_KEY 环境变量或传入 api_key 参数")
        self.model = SUPPORTED_MODELS.get(model.lower(), model)
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        
    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 4096, temperature: float = 0.3, timeout: int = 120, **kwargs) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, **kwargs}
        response = requests.post(self.chat_endpoint, headers=self._get_headers(), json=payload, timeout=timeout)
        if response.status_code != 200:
            raise Exception(f"API请求失败 [{response.status_code}]: {response.text}")
        return response.json()
    
    def complete(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.3, timeout: int = 120, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, max_tokens, temperature, timeout, **kwargs)
        if "choices" not in response:
            raise Exception(f"API响应异常: {response}")
        return response["choices"][0]["message"]["content"]

class SiliconFlowExtractor:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.client = SiliconFlowClient(api_key=api_key, model=model)
        self._fewshot_cache = {}
        self.fewshot_dir = Path(__file__).parent / "fewshot_samples"
        
    def _call_llm(self, prompt: str, max_tokens: int = 6000, timeout: int = 300) -> str:
        return self.client.complete(prompt, max_tokens=max_tokens, timeout=timeout)
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        try:
            from production_extractor_v79 import MedicalPDFExtractor as BaseExtractor
            class SFExtractor(BaseExtractor):
                def __init__(inner_self, client):
                    inner_self.api_url = None
                    inner_self._fewshot_cache = {}
                    inner_self._client = client
                def _call_llm(inner_self, prompt: str, max_tokens: int = 6000, timeout: int = 300) -> str:
                    return inner_self._client.complete(prompt, max_tokens=max_tokens, timeout=timeout)
            return SFExtractor(self.client).extract(pdf_path)
        except ImportError:
            return {"success": False, "error": "无法导入 production_extractor_v79"}

def test_connection(api_key: Optional[str] = None) -> bool:
    try:
        client = SiliconFlowClient(api_key=api_key)
        response = client.complete("你好，请回复'连接成功'", max_tokens=20, timeout=30)
        print(f"✓ 连接成功: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_connection()
    elif len(sys.argv) > 1 and sys.argv[1] == "models":
        print("支持的模型:", list(SUPPORTED_MODELS.keys()))
    elif len(sys.argv) > 1:
        result = SiliconFlowExtractor().extract(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("用法: python siliconflow_client.py [test|models|<pdf_path>]")
