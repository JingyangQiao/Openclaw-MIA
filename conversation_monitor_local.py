#!/usr/bin/env python3
"""
对话监控与保存器 - 本地文件保存版本

特性：
1. 严格单轮对话（不累积历史）
2. 每轮对话保存为独立的 jsonl 文件
3. 保存到指定文件夹，不再发送到 Server B
"""
import json
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from collections import OrderedDict
import threading
import os
from pathlib import Path

# 保存目录
SAVE_DIR = Path("./tmp/openclaw_high_value_chat")

# 会话超时时间（10分钟）
SESSION_TIMEOUT_SECONDS = 600

# 去重缓存最大条目数
MAX_PROCESSED_IDS = 10000


@dataclass
class Message:
    """单条消息"""
    role: str
    content: str
    timestamp: datetime
    message_id: str = ""


@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    turn_count: int = 0
    
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) - self.last_activity > __import__('datetime').timedelta(seconds=SESSION_TIMEOUT_SECONDS)
    
    def build_single_turn(self, user_msg: Message, assistant_msg: Message) -> Optional[Dict[str, Any]]:
        """构建严格单轮对话（不包含历史）"""
        self.turn_count += 1
        
        # 严格单轮：只包含当前 user + assistant
        turn_messages = [
            {"role": "user", "content": user_msg.content},
            {"role": "assistant", "content": assistant_msg.content}
        ]
        
        # 构建 prompt_text（只有 user 消息）
        prompt_text = f"<|im_start|>user\n{user_msg.content}<|im_end|>\n<|im_start|>assistant\n"
        
        # 构建 response_text
        response_text = assistant_msg.content.rstrip()
        if not response_text.endswith("<|im_end|>"):
            response_text += "<|im_end|>"
        response_text += "\n"
        
        return {
            "session_id": self.session_id,
            "turn": self.turn_count,
            "timestamp": user_msg.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": turn_messages,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": None,
            "next_state": None
        }


class LRUProcessedIds:
    """LRU 缓存"""
    
    def __init__(self, max_size: int = MAX_PROCESSED_IDS):
        self.max_size = max_size
        self._cache: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
    
    def add(self, message_id: str) -> bool:
        if not message_id:
            return True
        
        with self._lock:
            if message_id in self._cache:
                self._cache.move_to_end(message_id)
                return False
            else:
                self._cache[message_id] = None
                if len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
                return True


class ConversationMonitor:
    """对话监控器 - 本地保存版"""
    
    def __init__(self):
        self.active_session: Optional[ConversationSession] = None
        self.pending_user_msg: Optional[Message] = None
        self.processed_ids = LRUProcessedIds(max_size=MAX_PROCESSED_IDS)
        self.lock = threading.Lock()
        
        # 确保保存目录存在
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[Monitor] Save directory: {SAVE_DIR}")
    
    def process_incoming_message(self, role: str, content: str, 
                                  timestamp: Optional[datetime] = None,
                                  message_id: str = "") -> bool:
        """处理新消息"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # 去重检查
        is_new = self.processed_ids.add(message_id)
        if not is_new:
            print(f"[Monitor] Duplicate ignored: {message_id[:16] if message_id else 'N/A'}...")
            return False
        
        with self.lock:
            # 检查会话
            if self.active_session is None or self.active_session.is_expired():
                self.active_session = ConversationSession(
                    session_id=f"feishu-{uuid.uuid4().hex[:16]}"
                )
                self.pending_user_msg = None
                print(f"[Monitor] New session: {self.active_session.session_id}")
            
            self.active_session.last_activity = timestamp
            
            msg = Message(role=role, content=content, timestamp=timestamp, message_id=message_id)
            
            if role == "user":
                self.pending_user_msg = msg
                return True
                
            elif role == "assistant":
                if self.pending_user_msg is None:
                    print("[Monitor] Warning: Assistant without user message")
                    return False
                
                # 构建单轮对话
                turn_data = self.active_session.build_single_turn(
                    self.pending_user_msg, msg
                )
                
                self.pending_user_msg = None
                
                if turn_data is None:
                    return False
                
                # 保存到本地文件
                self._save_turn_to_file(turn_data)
                return True
        
        return True
    
    def _save_turn_to_file(self, turn_data: Dict[str, Any]):
        """将单轮对话保存为独立的 jsonl 文件"""
        try:
            # 构建文件名: {timestamp}_{session_id}_turn{turn}.jsonl
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = turn_data["session_id"]
            turn = turn_data["turn"]
            filename = f"{timestamp}_{session_id}_turn{turn}.jsonl"
            filepath = SAVE_DIR / filename
            
            # 写入 jsonl 格式（每行一个 JSON 对象）
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json.dumps(turn_data, ensure_ascii=False) + '\n')
            
            print(f"[Monitor] Saved turn #{turn} to {filename}")
            
        except Exception as e:
            print(f"[Monitor] Error saving turn: {e}")


# 全局实例
_monitor: Optional[ConversationMonitor] = None
_monitor_lock = threading.Lock()


def get_monitor() -> ConversationMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = ConversationMonitor()
    return _monitor


def process_message(role: str, content: str, 
                   timestamp: Optional[datetime] = None,
                   message_id: str = "") -> bool:
    monitor = get_monitor()
    return monitor.process_incoming_message(role, content, timestamp, message_id)


if __name__ == "__main__":
    print("Conversation Monitor (Local Save Version)")
    print("=" * 50)
    print(f"Save directory: {SAVE_DIR}")
    print("Test completed!")
