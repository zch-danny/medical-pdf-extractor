"""
WebSocket 进度推送模块

功能:
- WebSocket 连接管理
- 实时任务进度推送
- 心跳检测
- 任务订阅
"""

import asyncio
import json
import time
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


@dataclass
class WebSocketConnection:
    """WebSocket 连接信息"""
    websocket: WebSocket
    client_id: str
    connected_at: datetime = field(default_factory=datetime.now)
    subscribed_tasks: Set[str] = field(default_factory=set)
    last_ping: float = field(default_factory=time.time)


class ConnectionManager:
    """
    WebSocket 连接管理器
    
    功能:
    - 连接生命周期管理
    - 消息广播
    - 任务订阅
    """
    
    _instance: Optional["ConnectionManager"] = None
    
    def __init__(self):
        # 所有活跃连接: client_id -> connection
        self._connections: Dict[str, WebSocketConnection] = {}
        # 任务订阅: task_id -> set of client_ids
        self._task_subscribers: Dict[str, Set[str]] = {}
        # 锁
        self._lock = asyncio.Lock()
        
        logger.info("WebSocket ConnectionManager 初始化")
    
    @classmethod
    def get_instance(cls) -> "ConnectionManager":
        """获取单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        建立连接
        
        Args:
            websocket: WebSocket 对象
            client_id: 客户端 ID
        """
        await websocket.accept()
        
        async with self._lock:
            # 如果已存在，先断开旧连接
            if client_id in self._connections:
                old_conn = self._connections[client_id]
                try:
                    await old_conn.websocket.close()
                except Exception:
                    pass
            
            self._connections[client_id] = WebSocketConnection(
                websocket=websocket,
                client_id=client_id,
            )
        
        logger.info(f"WebSocket 连接建立: {client_id}")
        
        # 发送欢迎消息
        await self.send_personal(client_id, {
            "type": "connected",
            "client_id": client_id,
            "message": "连接成功",
        })
    
    async def disconnect(self, client_id: str) -> None:
        """
        断开连接
        
        Args:
            client_id: 客户端 ID
        """
        async with self._lock:
            if client_id in self._connections:
                conn = self._connections.pop(client_id)
                
                # 清理任务订阅
                for task_id in conn.subscribed_tasks:
                    if task_id in self._task_subscribers:
                        self._task_subscribers[task_id].discard(client_id)
                        if not self._task_subscribers[task_id]:
                            del self._task_subscribers[task_id]
        
        logger.info(f"WebSocket 连接断开: {client_id}")
    
    async def subscribe_task(self, client_id: str, task_id: str) -> bool:
        """
        订阅任务进度
        
        Args:
            client_id: 客户端 ID
            task_id: 任务 ID
            
        Returns:
            是否订阅成功
        """
        async with self._lock:
            if client_id not in self._connections:
                return False
            
            conn = self._connections[client_id]
            conn.subscribed_tasks.add(task_id)
            
            if task_id not in self._task_subscribers:
                self._task_subscribers[task_id] = set()
            self._task_subscribers[task_id].add(client_id)
        
        logger.debug(f"客户端 {client_id} 订阅任务 {task_id}")
        return True
    
    async def unsubscribe_task(self, client_id: str, task_id: str) -> None:
        """取消订阅任务"""
        async with self._lock:
            if client_id in self._connections:
                self._connections[client_id].subscribed_tasks.discard(task_id)
            
            if task_id in self._task_subscribers:
                self._task_subscribers[task_id].discard(client_id)
    
    async def send_personal(self, client_id: str, message: Dict[str, Any]) -> bool:
        """
        发送个人消息
        
        Args:
            client_id: 客户端 ID
            message: 消息内容
            
        Returns:
            是否发送成功
        """
        async with self._lock:
            if client_id not in self._connections:
                return False
            
            conn = self._connections[client_id]
        
        try:
            await conn.websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"发送消息失败: {client_id}, {e}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast_task_progress(
        self,
        task_id: str,
        progress: int,
        message: str,
        status: str = "processing",
        result: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        广播任务进度
        
        Args:
            task_id: 任务 ID
            progress: 进度 (0-100)
            message: 进度消息
            status: 任务状态
            result: 任务结果（完成时）
            
        Returns:
            发送成功的客户端数量
        """
        async with self._lock:
            subscribers = self._task_subscribers.get(task_id, set()).copy()
        
        if not subscribers:
            return 0
        
        payload = {
            "type": "task_progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        
        if result is not None:
            payload["result"] = result
        
        success_count = 0
        for client_id in subscribers:
            if await self.send_personal(client_id, payload):
                success_count += 1
        
        return success_count
    
    async def broadcast_all(self, message: Dict[str, Any]) -> int:
        """
        广播给所有连接
        
        Args:
            message: 消息内容
            
        Returns:
            发送成功的数量
        """
        async with self._lock:
            client_ids = list(self._connections.keys())
        
        success_count = 0
        for client_id in client_ids:
            if await self.send_personal(client_id, message):
                success_count += 1
        
        return success_count
    
    async def handle_message(self, client_id: str, message: str) -> None:
        """
        处理客户端消息
        
        支持的消息类型:
        - ping: 心跳
        - subscribe: 订阅任务
        - unsubscribe: 取消订阅
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "ping":
                # 心跳响应
                async with self._lock:
                    if client_id in self._connections:
                        self._connections[client_id].last_ping = time.time()
                
                await self.send_personal(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat(),
                })
            
            elif msg_type == "subscribe":
                # 订阅任务
                task_id = data.get("task_id")
                if task_id:
                    success = await self.subscribe_task(client_id, task_id)
                    await self.send_personal(client_id, {
                        "type": "subscribed",
                        "task_id": task_id,
                        "success": success,
                    })
            
            elif msg_type == "unsubscribe":
                # 取消订阅
                task_id = data.get("task_id")
                if task_id:
                    await self.unsubscribe_task(client_id, task_id)
                    await self.send_personal(client_id, {
                        "type": "unsubscribed",
                        "task_id": task_id,
                    })
            
            else:
                logger.warning(f"未知消息类型: {msg_type}")
        
        except json.JSONDecodeError:
            logger.warning(f"无效的 JSON 消息: {message[:100]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "active_connections": len(self._connections),
            "subscribed_tasks": len(self._task_subscribers),
            "clients": list(self._connections.keys()),
        }


# 全局连接管理器
manager = ConnectionManager.get_instance()


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket 端点处理函数
    
    使用方法 (在 FastAPI 路由中):
        @app.websocket("/ws/{client_id}")
        async def websocket_route(websocket: WebSocket, client_id: str):
            await websocket_endpoint(websocket, client_id)
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # 等待客户端消息
            data = await websocket.receive_text()
            await manager.handle_message(client_id, data)
    
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket 错误: {client_id}, {e}")
        await manager.disconnect(client_id)


# ===========================================
# 进度推送辅助函数
# ===========================================

async def push_task_progress(
    task_id: str,
    progress: int,
    message: str,
    status: str = "processing",
    result: Optional[Dict[str, Any]] = None,
) -> int:
    """
    推送任务进度（便捷函数）
    
    Args:
        task_id: 任务 ID
        progress: 进度 (0-100)
        message: 进度消息
        status: 任务状态
        result: 任务结果
        
    Returns:
        发送成功的客户端数量
    """
    return await manager.broadcast_task_progress(
        task_id=task_id,
        progress=progress,
        message=message,
        status=status,
        result=result,
    )


def create_progress_callback(task_id: str):
    """
    创建进度回调函数
    
    用于集成到摘要服务中
    
    Args:
        task_id: 任务 ID
        
    Returns:
        异步进度回调函数
    """
    async def callback(progress: int, message: str):
        await push_task_progress(task_id, progress, message)
    
    return callback
