"""
服务器A - OpenClaw桥接服务 (端口6333)
功能：
1. 收集MIA Planner和OpenClaw的对话数据
2. 实时传输到服务器B
3. 接收训练好的权重
"""
from flask import Flask, request, jsonify
import requests
import threading
import queue
from datetime import datetime
import json
from pathlib import Path
import uuid

app = Flask(__name__)

# ========== 配置 ==========
# 服务器B的代理URL
import os
SERVER_B_URL = os.environ.get(
    "SERVER_B_URL",
    ""
)

# 本地存储
WEIGHT_DIR = Path("./weights")
WEIGHT_DIR.mkdir(exist_ok=True)

CONVERSATION_DIR = Path("./conversations")
CONVERSATION_DIR.mkdir(exist_ok=True)

# ========== 数据队列 ==========
send_queue = queue.Queue()

# 当前会话状态
active_sessions = {}  # session_id -> session_data

# ========== 后台发送线程 ==========
def send_worker():
    """后台线程：将数据发送到服务器B"""
    while True:
        try:
            data = send_queue.get(timeout=1)
            try:
                resp = requests.post(
                    f"{SERVER_B_URL}/receive",
                    json=data,
                    timeout=10
                )
                if resp.status_code == 200:
                    print(f"[A] Sent to B: {data.get('session_id', 'unknown')} -> {resp.status_code}")
                else:
                    print(f"[A] Failed to send: {resp.status_code} - {resp.text[:100]}")
            except Exception as e:
                print(f"[A] Send error: {e}")
        except queue.Empty:
            continue

# 启动后台线程
threading.Thread(target=send_worker, daemon=True).start()

# ========== API端点 ==========
@app.route('/')
def index():
    return jsonify({
        "service": "OpenClaw Bridge Server A",
        "server_b": SERVER_B_URL,
        "endpoints": {
            "health": "/health",
            "collect": "/collect (POST) - 收集训练数据",
            "conversation/start": "/conversation/start (POST) - 开始新会话",
            "conversation/turn": "/conversation/turn (POST) - 记录对话轮次",
            "conversation/end": "/conversation/end (POST) - 结束会话",
            "upload_weight": "/upload_weight (POST) - 接收权重并自动热更新SGLang",
            "update_sglang": "/update_sglang (POST) - 手动触发SGLang热更新",
            "sglang_status": "/sglang_status (GET) - 获取SGLang状态"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "server": "A",
        "queue_size": send_queue.qsize(),
        "active_sessions": len(active_sessions),
        "time": datetime.now().isoformat()
    })

@app.route('/collect', methods=['POST'])
def collect():
    """通用数据收集接口"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    data["timestamp"] = datetime.now().isoformat()
    data["server"] = "A"
    send_queue.put(data)
    
    return jsonify({
        "status": "queued",
        "queue_size": send_queue.qsize()
    })

# ========== 对话收集接口 ==========
@app.route('/conversation/start', methods=['POST'])
def conversation_start():
    """开始一个新的对话会话"""
    data = request.get_json() or {}
    
    session_id = data.get('session_id') or str(uuid.uuid4())
    
    session_data = {
        "session_id": session_id,
        "start_time": datetime.now().isoformat(),
        "user_question": data.get('question', ''),
        "source": data.get('source', 'unknown'),
        "turns": [],
        "status": "active"
    }
    
    active_sessions[session_id] = session_data
    
    print(f"[A] Conversation started: {session_id}")
    
    return jsonify({
        "status": "started",
        "session_id": session_id
    })

@app.route('/conversation/turn', methods=['POST'])
def conversation_turn():
    """记录对话的一个轮次"""
    data = request.get_json() or {}
    
    session_id = data.get('session_id')
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    
    turn_data = {
        "turn_number": len(active_sessions[session_id]['turns']) + 1,
        "timestamp": datetime.now().isoformat(),
        "role": data.get('role', 'unknown'),  # 'user', 'assistant', 'planner', 'tool'
        "content": data.get('content', ''),
        "metadata": data.get('metadata', {})
    }
    
    active_sessions[session_id]['turns'].append(turn_data)
    
    # 实时发送到B（可选，也可以等会话结束再发）
    realtime_data = {
        "type": "conversation_turn",
        "session_id": session_id,
        "turn": turn_data,
        "timestamp": datetime.now().isoformat()
    }
    send_queue.put(realtime_data)
    
    print(f"[A] Turn recorded: {session_id}, turn={turn_data['turn_number']}")
    
    return jsonify({
        "status": "recorded",
        "session_id": session_id,
        "turn_number": turn_data['turn_number']
    })

@app.route('/conversation/end', methods=['POST'])
def conversation_end():
    """结束会话并发送完整数据"""
    data = request.get_json() or {}
    
    session_id = data.get('session_id')
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    
    session_data = active_sessions[session_id]
    session_data['end_time'] = datetime.now().isoformat()
    session_data['status'] = 'completed'
    session_data['success'] = data.get('success', False)
    session_data['final_answer'] = data.get('final_answer', '')
    session_data['execution_time'] = data.get('execution_time', 0)
    
    # 保存到本地
    conv_file = CONVERSATION_DIR / f"{session_id}.json"
    with open(conv_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    # 发送到B（完整会话数据）
    send_data = {
        "type": "conversation_complete",
        "session_id": session_id,
        "data": session_data,
        "timestamp": datetime.now().isoformat()
    }
    send_queue.put(send_data)
    
    # 清理内存
    del active_sessions[session_id]
    
    print(f"[A] Conversation ended: {session_id}, turns={len(session_data['turns'])}")
    
    return jsonify({
        "status": "completed",
        "session_id": session_id,
        "saved_to": str(conv_file)
    })

@app.route('/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    """获取会话数据"""
    if session_id in active_sessions:
        return jsonify(active_sessions[session_id])
    
    conv_file = CONVERSATION_DIR / f"{session_id}.json"
    if conv_file.exists():
        with open(conv_file, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    
    return jsonify({"error": "Session not found"}), 404

# ========== 权重接收 ==========
@app.route('/upload_weight', methods=['POST'])
def upload_weight():
    """接收训练好的权重文件"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"model_{timestamp}.tar.gz"
    save_path = WEIGHT_DIR / filename
    
    file.save(save_path)
    file_size = save_path.stat().st_size
    
    print(f"[A] Received weight: {filename} ({file_size} bytes)")
    
    # 触发 SGLang 热更新（在后台线程中执行，避免阻塞请求）
    def update_sglang_async():
        import time
        time.sleep(2)  # 等待文件写入完成
        try:
            import subprocess
            result = subprocess.run(
                ["bash", str(Path(__file__).parent / "sglang_manager.sh"), "hot_update"],
                capture_output=True,
                text=True,
                timeout=300
            )
            print(f"[A] SGLang hot update result: {result.returncode}")
            print(f"[A] stdout: {result.stdout}")
            if result.stderr:
                print(f"[A] stderr: {result.stderr}")
        except Exception as e:
            print(f"[A] Failed to hot update SGLang: {e}")
    
    threading.Thread(target=update_sglang_async, daemon=True).start()
    
    return jsonify({
        "status": "success",
        "filename": filename,
        "size_bytes": file_size,
        "saved_to": str(save_path),
        "message": "Weight received. SGLang will be hot-updated automatically (no restart)."
    })

@app.route('/update_sglang', methods=['POST'])
def update_sglang():
    """手动触发 SGLang 热更新"""
    try:
        import subprocess
        result = subprocess.run(
            ["bash", str(Path(__file__).parent / "sglang_manager.sh"), "hot_update"],
            capture_output=True,
            text=True,
            timeout=300
        )
        return jsonify({
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/sglang_status', methods=['GET'])
def sglang_status():
    """获取 SGLang 状态"""
    try:
        import subprocess
        result = subprocess.run(
            ["bash", str(Path(__file__).parent / "sglang_manager.sh"), "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return jsonify({
            "status": "success",
            "output": result.stdout
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ========== 便捷函数（供其他模块调用）==========
def send_conversation_data(session_id: str, question: str, plan: str = '', 
                          workflow: list = None, success: bool = False):
    """便捷函数：发送完整的对话数据"""
    data = {
        "type": "conversation_complete",
        "session_id": session_id or str(uuid.uuid4()),
        "data": {
            "user_question": question,
            "plan": plan,
            "workflow": workflow or [],
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
    }
    send_queue.put(data)

def record_turn(session_id: str, role: str, content: str, metadata: dict = None):
    """便捷函数：记录单轮对话"""
    data = {
        "type": "conversation_turn",
        "session_id": session_id,
        "turn": {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
    }
    send_queue.put(data)

if __name__ == '__main__':
    print("[A] Starting OpenClaw Bridge Server on port 6333")
    print(f"[A] Will send data to B: {SERVER_B_URL}")
    print(f"[A] Conversations will be saved to: {CONVERSATION_DIR}")
    print(f"[A] Weights will be saved to: {WEIGHT_DIR}")
    app.run(host='0.0.0.0', port=6333, threaded=True)
