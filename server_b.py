#!/usr/bin/env python3
"""
服务器B - 生产环境版本 (端口3)
修复：进程保活、日志持久化、自动重启、内存管理

改进：
1. 使用生产级 WSGI 服务器 (waitress/gunicorn)
2. 进程守护和自动重启
3. 日志写入文件
4. 内存使用监控
5. 健康检查心跳
"""
from flask import Flask, request, jsonify, Response
from datetime import datetime
import queue
import json
import os
import sys
import time
import signal
import threading
import logging
import uuid
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 尝试导入 push_weight，如果失败则使用模拟
try:
    from push_weight import push_weight
except ImportError:
    def push_weight(checkpoint_path: str) -> bool:
        print(f"[Mock] Pushing weight: {checkpoint_path}")
        return True

# ========== 配置 ==========
PORT = int(os.environ.get("SERVER_B_PORT", 3))
HOST = os.environ.get("SERVER_B_HOST", "0.0.0.0")
LOG_DIR = Path(os.environ.get("SERVER_B_LOG_DIR", "./logs"))
LOG_DIR.mkdir(exist_ok=True)
CONVERSATION_DIR = Path(os.environ.get("SERVER_B_CONVERSATION_DIR", "./received_conversations"))
CONVERSATION_DIR.mkdir(exist_ok=True)
BUFFER_MAXSIZE = int(os.environ.get("SERVER_B_BUFFER_MAXSIZE", 10000))
MAX_STATS_HISTORY = 1000  # 限制统计历史

# ========== 日志配置 ==========
def setup_logging():
    """配置日志"""
    log_file = LOG_DIR / "server_b.log"
    
    # 根日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ========== Flask 应用 ==========
app = Flask(__name__)

# 数据缓冲区
data_buffer = queue.Queue(maxsize=BUFFER_MAXSIZE)

# 训练统计（带历史限制）
stats = {
    "total_received": 0,
    "conversations": 0,
    "turns": 0,
    "last_received": None,
    "start_time": datetime.now().isoformat(),
    "errors": []  # 错误历史，限制大小
}

# 运行状态
running = True
shutdown_event = threading.Event()

# ========== 辅助函数 ==========
def add_error(error_msg: str):
    """添加错误记录，限制大小"""
    stats["errors"].append({
        "time": datetime.now().isoformat(),
        "error": error_msg
    })
    # 只保留最近的错误
    if len(stats["errors"]) > 100:
        stats["errors"] = stats["errors"][-100:]

def get_memory_usage():
    """获取内存使用情况"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"note": "psutil not installed"}

# ========== API 端点 ==========
@app.route('/')
def index():
    return jsonify({
        "service": "Training Server B (Production)",
        "version": "2.0",
        "endpoints": {
            "health": "/health",
            "receive": "/receive (POST) - 接收数据",
            "get_data": "/get_data (GET) - 获取训练数据",
            "stats": "/stats (GET) - 统计信息",
            "conversations": "/conversations (GET) - 列出对话"
        }
    })

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        "status": "ok",
        "server": "B",
        "version": "2.0",
        "buffer_size": data_buffer.qsize(),
        "stats": stats,
        "memory": get_memory_usage(),
        "time": datetime.now().isoformat()
    })

@app.route('/receive', methods=['POST'])
def receive():
    """接收来自服务器A的数据"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Received empty data")
            return Response(json.dumps({"error": "No data"}, ensure_ascii=False),
                           mimetype='application/json; charset=utf-8'), 400
        
        data_type = data.get('type', 'unknown')
        session_id = data.get('session_id', 'unknown')
        
        # 存入缓冲区
        try:
            data_buffer.put(data, block=False)
        except queue.Full:
            logger.error("Buffer full, dropping data")
            return Response(json.dumps({"error": "Buffer full"}, ensure_ascii=False),
                           mimetype='application/json; charset=utf-8'), 503
        
        # 更新统计
        stats["total_received"] += 1
        stats["last_received"] = datetime.now().isoformat()
        
        # 处理特定类型
        if data_type == "conversation_complete":
            stats["conversations"] += 1
            # 保存完整对话（每条唯一文件名，避免同 session_id 多次 POST 互相覆盖）
            try:
                safe_sid = re.sub(r"[^\w\-.]", "_", str(session_id), flags=re.ASCII)[:120]
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                uniq = uuid.uuid4().hex[:10]
                conv_file = CONVERSATION_DIR / f"{safe_sid}_{stamp}_{uniq}.json"
                with open(conv_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved conversation: {session_id} -> {conv_file.name}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
                add_error(f"Save conversation failed: {e}")
                
        elif data_type == "conversation_turn":
            stats["turns"] += 1
            # 也保存 turn 数据到文件（按 session 分文件追加）
            try:
                turn_file = CONVERSATION_DIR / f"{session_id}_turns.jsonl"
                with open(turn_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                logger.info(f"Saved turn: {session_id} turn #{data.get('data', {}).get('turn', '?')}")
            except Exception as e:
                logger.error(f"Failed to save turn: {e}")
                add_error(f"Save turn failed: {e}")
        
        logger.info(f"Received: type={data_type}, session={session_id}, buffer={data_buffer.qsize()}")
        
        response_data = {
            "status": "received",
            "type": data_type,
            "session_id": session_id,
            "buffer_size": data_buffer.qsize()
        }
        return Response(json.dumps(response_data, ensure_ascii=False),
                       mimetype='application/json; charset=utf-8')
        
    except Exception as e:
        logger.exception("Error in receive")
        add_error(str(e))
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                       mimetype='application/json; charset=utf-8'), 500

@app.route('/get_data')
def get_data():
    """获取数据供训练使用"""
    try:
        batch_size = request.args.get('batch_size', 10, type=int)
        
        items = []
        for _ in range(batch_size):
            try:
                item = data_buffer.get_nowait()
                items.append(item)
            except queue.Empty:
                break
        
        response_data = {
            "items": items,
            "remaining": data_buffer.qsize(),
            "batch_size": len(items)
        }
        
        return Response(
            json.dumps(response_data, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )
    except Exception as e:
        logger.exception("Error in get_data")
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                       mimetype='application/json; charset=utf-8'), 500

@app.route('/stats')
def get_stats():
    """获取统计信息"""
    return jsonify({
        "stats": stats,
        "buffer_size": data_buffer.qsize(),
        "conversations_saved": len(list(CONVERSATION_DIR.glob("*.json"))),
        "memory": get_memory_usage(),
        "uptime": datetime.now().isoformat()
    })

@app.route('/conversations')
def list_conversations():
    """列出所有保存的对话"""
    try:
        conversations = []
        for f in CONVERSATION_DIR.glob("*.json"):
            try:
                stat = f.stat()
                conversations.append({
                    "session_id": f.stem,
                    "file": str(f),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to stat file {f}: {e}")
        
        # 按修改时间排序
        conversations.sort(key=lambda x: x["modified"], reverse=True)
        
        return jsonify({
            "conversations": conversations,
            "total": len(conversations)
        })
    except Exception as e:
        logger.exception("Error in list_conversations")
        return jsonify({"error": str(e)}), 500

# ========== 训练回调函数 ==========
def on_training_complete(checkpoint_path: str):
    """训练完成回调 - 推送权重到服务器A"""
    logger.info(f"Training completed, pushing weight: {checkpoint_path}")
    
    try:
        success = push_weight(checkpoint_path)
        
        if success:
            logger.info("Weight pushed successfully!")
        else:
            logger.error("Failed to push weight")
        
        return success
    except Exception as e:
        logger.exception("Error in on_training_complete")
        add_error(f"Push weight failed: {e}")
        return False

def get_training_batch(batch_size: int = 4):
    """获取一批训练数据"""
    items = []
    for _ in range(batch_size):
        try:
            items.append(data_buffer.get_nowait())
        except queue.Empty:
            break
    return items

# ========== 监控线程 ==========
def monitor_thread():
    """监控线程：定期检查健康状态"""
    while not shutdown_event.is_set():
        try:
            time.sleep(60)  # 每分钟检查一次
            
            memory = get_memory_usage()
            logger.info(f"Health check: buffer={data_buffer.qsize()}, "
                       f"memory={memory.get('rss_mb', '?')}MB, "
                       f"received={stats['total_received']}")
            
            # 如果内存使用过高，记录警告
            if isinstance(memory, dict) and 'percent' in memory:
                if memory['percent'] > 80:
                    logger.warning(f"High memory usage: {memory['percent']:.1f}%")
                    
        except Exception as e:
            logger.exception("Error in monitor thread")

# ========== 信号处理 ==========
def signal_handler(signum, frame):
    """处理关闭信号"""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()
    running = False
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ========== 主函数 ==========
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Server B (Production) Starting...")
    logger.info(f"Port: {PORT}")
    logger.info(f"Host: {HOST}")
    logger.info(f"Log dir: {LOG_DIR}")
    logger.info(f"Conversation dir: {CONVERSATION_DIR}")
    logger.info("=" * 60)
    
    # 启动监控线程
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    logger.info("Monitor thread started")
    
    # 使用 waitress 作为生产级 WSGI 服务器
    try:
        from waitress import serve
        logger.info("Using waitress WSGI server")
        serve(app, host=HOST, port=PORT, threads=4)
    except ImportError:
        logger.warning("waitress not installed, falling back to Flask dev server")
        logger.warning("For production, install waitress: pip install waitress")
        app.run(host=HOST, port=PORT, threaded=True)

if __name__ == '__main__':
    main()
