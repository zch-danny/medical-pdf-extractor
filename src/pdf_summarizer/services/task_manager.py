"""
异步任务管理器

功能:
1. 任务提交与后台处理
2. 任务状态查询
3. 任务进度跟踪
4. 任务结果存储
"""
import asyncio
import uuid
import time
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock
from loguru import logger


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"        # 失败


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0  # 0-100
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    异步任务管理器
    
    支持:
    - 任务提交与后台执行
    - 任务状态查询
    - 任务进度更新
    - 自动清理过期任务
    """
    
    _instance: Optional["TaskManager"] = None
    
    def __init__(
        self,
        max_concurrent: int = 5,
        max_tasks: int = 1000,
        task_ttl_hours: int = 24,
    ):
        self.max_concurrent = max_concurrent
        self.max_tasks = max_tasks
        self.task_ttl = timedelta(hours=task_ttl_hours)
        
        self._tasks: OrderedDict[str, TaskInfo] = OrderedDict()
        self._lock = Lock()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._running = False
        
        logger.info(f"TaskManager 初始化: max_concurrent={max_concurrent}")
    
    @classmethod
    def get_instance(cls) -> "TaskManager":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_semaphore(self) -> asyncio.Semaphore:
        """确保信号量存在"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore
    
    async def submit(
        self,
        task_fn: Callable[["TaskInfo"], Awaitable[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交异步任务
        
        Args:
            task_fn: 任务函数，接收 TaskInfo 用于更新进度
            metadata: 任务元数据
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        with self._lock:
            # 清理过期任务
            self._cleanup_expired()
            
            # 检查任务数量限制
            if len(self._tasks) >= self.max_tasks:
                # 删除最老的已完成任务
                self._evict_oldest_completed()
            
            self._tasks[task_id] = task_info
        
        # 启动后台任务
        asyncio.create_task(self._run_task(task_id, task_fn))
        
        logger.info(f"任务已提交: {task_id}")
        return task_id
    
    async def _run_task(
        self,
        task_id: str,
        task_fn: Callable[["TaskInfo"], Awaitable[Dict[str, Any]]],
    ) -> None:
        """执行任务"""
        semaphore = self._ensure_semaphore()
        
        async with semaphore:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return
            
            # 更新状态为处理中
            task_info.status = TaskStatus.PROCESSING
            task_info.started_at = datetime.now()
            task_info.message = "正在处理..."
            
            try:
                # 执行任务
                result = await task_fn(task_info)
                
                # 更新为完成
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.now()
                task_info.progress = 100
                task_info.message = "处理完成"
                task_info.result = result
                
                logger.info(f"任务完成: {task_id}")
                
            except Exception as e:
                # 更新为失败
                task_info.status = TaskStatus.FAILED
                task_info.completed_at = datetime.now()
                task_info.error = str(e)
                task_info.message = f"处理失败: {e}"
                
                logger.error(f"任务失败: {task_id}, 错误: {e}")
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务信息"""
        return self._tasks.get(task_id)
    
    def update_progress(self, task_id: str, progress: int, message: str = "") -> None:
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度 0-100
            message: 进度消息
        """
        task_info = self._tasks.get(task_id)
        if task_info:
            task_info.progress = min(100, max(0, progress))
            if message:
                task_info.message = message
    
    def _cleanup_expired(self) -> None:
        """清理过期任务"""
        now = datetime.now()
        expired = [
            tid for tid, task in self._tasks.items()
            if task.completed_at and now - task.completed_at > self.task_ttl
        ]
        for tid in expired:
            del self._tasks[tid]
        
        if expired:
            logger.debug(f"清理过期任务: {len(expired)} 个")
    
    def _evict_oldest_completed(self) -> None:
        """淘汰最老的已完成任务"""
        for tid, task in list(self._tasks.items()):
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                del self._tasks[tid]
                return
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            status_counts = {s.value: 0 for s in TaskStatus}
            for task in self._tasks.values():
                status_counts[task.status.value] += 1
            
            return {
                "total_tasks": len(self._tasks),
                "status_counts": status_counts,
                "max_concurrent": self.max_concurrent,
            }
