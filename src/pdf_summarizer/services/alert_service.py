"""
告警服务

功能:
1. 错误率监控
2. 延迟监控
3. Webhook 告警

参考 safety_llm_deploy 的告警机制
"""
import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from loguru import logger
import httpx

try:
    from pdf_summarizer.core.high_performance import ALERT_CONFIG
except ImportError:
    ALERT_CONFIG = {
        "enabled": False,
        "webhook_url": "",
        "error_rate_threshold": 0.1,
        "latency_threshold_ms": 30000,
        "queue_size_threshold": 50,
    }


@dataclass
class AlertEvent:
    """告警事件"""
    level: str  # info, warning, error, critical
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MetricWindow:
    """指标滑动窗口"""
    window_size: int = 100  # 最近 N 个请求
    errors: deque = field(default_factory=lambda: deque(maxlen=100))
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_request(self, success: bool, latency_ms: float) -> None:
        """记录请求"""
        self.errors.append(0 if success else 1)
        self.latencies.append(latency_ms)
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if not self.errors:
            return 0.0
        return sum(self.errors) / len(self.errors)
    
    @property
    def avg_latency_ms(self) -> float:
        """平均延迟"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    @property
    def p99_latency_ms(self) -> float:
        """P99 延迟"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class AlertService:
    """
    告警服务
    
    功能:
    - 监控错误率、延迟
    - 触发 Webhook 告警
    - 告警去重（冷却时间）
    """
    
    _instance: Optional["AlertService"] = None
    
    def __init__(self):
        self.enabled = ALERT_CONFIG.get("enabled", False)
        self.webhook_url = ALERT_CONFIG.get("webhook_url", "")
        self.error_rate_threshold = ALERT_CONFIG.get("error_rate_threshold", 0.1)
        self.latency_threshold_ms = ALERT_CONFIG.get("latency_threshold_ms", 30000)
        
        self._metrics = MetricWindow()
        self._recent_alerts: Dict[str, datetime] = {}  # 告警去重
        self._alert_cooldown = timedelta(minutes=5)  # 同类告警冷却时间
        self._http_client: Optional[httpx.AsyncClient] = None
        
        if self.enabled:
            logger.info(f"AlertService 已启用: webhook={self.webhook_url[:30]}...")
        else:
            logger.info("AlertService 未启用")
    
    @classmethod
    def get_instance(cls) -> "AlertService":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def record_request(self, success: bool, latency_ms: float) -> None:
        """
        记录请求结果
        
        Args:
            success: 是否成功
            latency_ms: 延迟（毫秒）
        """
        self._metrics.record_request(success, latency_ms)
        
        if not self.enabled:
            return
        
        # 检查是否需要告警
        asyncio.create_task(self._check_alerts())
    
    async def _check_alerts(self) -> None:
        """检查并触发告警"""
        now = datetime.now()
        
        # 错误率告警
        if self._metrics.error_rate > self.error_rate_threshold:
            await self._maybe_alert(
                key="high_error_rate",
                event=AlertEvent(
                    level="error",
                    title="错误率过高",
                    message=f"当前错误率 {self._metrics.error_rate:.1%} 超过阈值 {self.error_rate_threshold:.1%}",
                    metadata={
                        "error_rate": self._metrics.error_rate,
                        "threshold": self.error_rate_threshold,
                    }
                )
            )
        
        # 延迟告警
        if self._metrics.p99_latency_ms > self.latency_threshold_ms:
            await self._maybe_alert(
                key="high_latency",
                event=AlertEvent(
                    level="warning",
                    title="延迟过高",
                    message=f"P99延迟 {self._metrics.p99_latency_ms:.0f}ms 超过阈值 {self.latency_threshold_ms}ms",
                    metadata={
                        "p99_latency_ms": self._metrics.p99_latency_ms,
                        "avg_latency_ms": self._metrics.avg_latency_ms,
                        "threshold_ms": self.latency_threshold_ms,
                    }
                )
            )
    
    async def _maybe_alert(self, key: str, event: AlertEvent) -> None:
        """
        可能触发告警（带去重）
        
        Args:
            key: 告警键（用于去重）
            event: 告警事件
        """
        now = datetime.now()
        
        # 检查冷却时间
        last_alert = self._recent_alerts.get(key)
        if last_alert and now - last_alert < self._alert_cooldown:
            return
        
        # 记录告警时间
        self._recent_alerts[key] = now
        
        # 发送告警
        await self._send_webhook(event)
    
    async def _send_webhook(self, event: AlertEvent) -> None:
        """发送 Webhook 告警"""
        if not self.webhook_url:
            logger.warning(f"告警未发送（无 webhook URL）: {event.title}")
            return
        
        try:
            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=10.0)
            
            payload = {
                "service": "PDF Summarization Service",
                "alert": event.to_dict(),
            }
            
            response = await self._http_client.post(self.webhook_url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"告警已发送: {event.title}")
            else:
                logger.warning(f"告警发送失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"告警发送异常: {e}")
    
    def get_metrics(self) -> dict:
        """获取当前指标"""
        return {
            "error_rate": f"{self._metrics.error_rate:.2%}",
            "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
            "p99_latency_ms": round(self._metrics.p99_latency_ms, 2),
            "sample_size": len(self._metrics.errors),
        }
    
    async def close(self) -> None:
        """关闭资源"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
