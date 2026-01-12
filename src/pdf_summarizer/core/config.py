"""
配置管理模块

使用 pydantic-settings 管理环境变量配置
支持配置热更新
"""

import threading
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置"""
    
    # === 环境 ===
    app_env: str = Field(default="prod", alias="APP_ENV")
    
    # === API 服务 ===
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    api_workers: int = Field(default=4, alias="API_WORKERS")
    api_debug: bool = Field(default=False, alias="API_DEBUG")
    
    # === 模型配置 ===
    model_name: str = Field(default="Qwen/Qwen3-8B", alias="MODEL_NAME")
    model_max_tokens: int = Field(default=2048, alias="MODEL_MAX_TOKENS")
    model_temperature: float = Field(default=0.3, alias="MODEL_TEMPERATURE")
    
    # === vLLM 配置 ===
    vllm_api_base: str = Field(default="http://localhost:8000/v1", alias="VLLM_API_BASE")
    vllm_api_key: str = Field(default="EMPTY", alias="VLLM_API_KEY")
    
    # === PDF 处理 ===
    max_pdf_size_mb: int = Field(default=50, alias="MAX_PDF_SIZE_MB")
    max_pdf_pages: int = Field(default=200, alias="MAX_PDF_PAGES")
    chunk_size: int = Field(default=8000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=500, alias="CHUNK_OVERLAP")
    
    # === 分块模式 (Qwen3-8B 优化) ===
    use_token_mode: bool = Field(default=True, alias="USE_TOKEN_MODE")  # 使用 token 模式分块
    max_chunk_tokens: int = Field(default=12000, alias="MAX_CHUNK_TOKENS")  # 每块最大 tokens
    
    # === 摘要配置 ===
    default_summary_language: str = Field(default="zh", alias="DEFAULT_SUMMARY_LANGUAGE")
    default_max_summary_length: int = Field(default=500, alias="DEFAULT_MAX_SUMMARY_LENGTH")
    min_summary_length: int = Field(default=100, alias="MIN_SUMMARY_LENGTH")
    
    # === 日志 ===
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    
    # === CORS ===
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        alias="CORS_ORIGINS"
    )
    
    # === 缓存 ===
    cache_enabled: bool = Field(default=False, alias="CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    
    # === OCR 配置 (DeepSeek-OCR) ===
    ocr_enabled: bool = Field(default=True, alias="OCR_ENABLED")
    ocr_api_base: str = Field(default="http://localhost:8001/v1", alias="OCR_API_BASE")
    ocr_model_name: str = Field(default="deepseek-ai/DeepSeek-OCR", alias="OCR_MODEL_NAME")
    ocr_dpi: int = Field(default=150, alias="OCR_DPI")
    text_coverage_threshold: float = Field(default=0.3, alias="TEXT_COVERAGE_THRESHOLD")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """解析 CORS 配置为列表"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def max_pdf_size_bytes(self) -> int:
        """PDF 最大字节数"""
        return self.max_pdf_size_mb * 1024 * 1024
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "protected_namespaces": ("settings_",),  # 避免 model_ 前缀警告
    }


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


class ConfigManager:
    """
    配置管理器
    
    支持配置热更新、配置变更回调
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._settings: Settings = Settings()
        self._callbacks: List[Callable[[Settings, Settings], None]] = []
        self._last_reload_time: Optional[datetime] = None
        self._reload_lock = threading.Lock()
        self._initialized = True
    
    @property
    def settings(self) -> Settings:
        """获取当前配置"""
        return self._settings
    
    @property
    def last_reload_time(self) -> Optional[datetime]:
        """上次重载时间"""
        return self._last_reload_time
    
    def reload(self) -> Dict[str, Any]:
        """
        重新加载配置
        
        Returns:
            配置变更信息
        """
        with self._reload_lock:
            old_settings = self._settings
            
            # 清除缓存并重新加载
            get_settings.cache_clear()
            new_settings = Settings()
            
            # 检测变更
            changes = self._detect_changes(old_settings, new_settings)
            
            # 更新配置
            self._settings = new_settings
            self._last_reload_time = datetime.now()
            
            # 触发回调
            if changes:
                for callback in self._callbacks:
                    try:
                        callback(old_settings, new_settings)
                    except Exception as e:
                        pass  # 忽略回调错误
            
            return {
                "success": True,
                "reload_time": self._last_reload_time.isoformat(),
                "changes": changes,
            }
    
    def _detect_changes(self, old: Settings, new: Settings) -> Dict[str, Dict[str, Any]]:
        """检测配置变更"""
        changes = {}
        
        # 可热更新的配置项
        hot_reload_fields = [
            "log_level",
            "model_temperature",
            "model_max_tokens",
            "cache_enabled",
            "cache_ttl_seconds",
            "default_max_summary_length",
        ]
        
        for field in hot_reload_fields:
            old_value = getattr(old, field, None)
            new_value = getattr(new, field, None)
            if old_value != new_value:
                changes[field] = {
                    "old": old_value,
                    "new": new_value,
                }
        
        return changes
    
    def register_callback(self, callback: Callable[[Settings, Settings], None]):
        """
        注册配置变更回调
        
        Args:
            callback: 回调函数，接收 (old_settings, new_settings) 参数
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Settings, Settings], None]):
        """取消注册回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """获取运行时配置信息"""
        return {
            "app_env": self._settings.app_env,
            "model_name": self._settings.model_name,
            "model_temperature": self._settings.model_temperature,
            "model_max_tokens": self._settings.model_max_tokens,
            "log_level": self._settings.log_level,
            "cache_enabled": self._settings.cache_enabled,
            "last_reload_time": self._last_reload_time.isoformat() if self._last_reload_time else None,
        }


# 配置管理器单例
config_manager = ConfigManager()

# 全局配置实例（向后兼容）
settings = config_manager.settings


def reload_settings() -> Dict[str, Any]:
    """重新加载配置（便捷函数）"""
    return config_manager.reload()
