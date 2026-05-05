#!/usr/bin/env python3
"""
对话收集服务 - 后台运行，通过文件接收数据

启动: python3 collect_service.py start
停止: python3 collect_service.py stop
状态: python3 collect_service.py status
发送: python3 collect_service.py send 'user msg' 'assistant reply'
"""
import json
import os
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conversation_monitor_local import process_message, get_monitor

PID_FILE = Path("./tmp/openclaw_collect_service.pid")
DATA_FILE = Path("./tmp/openclaw_collect.jsonl")
LOG_FILE = Path("./tmp/openclaw_collect_service.log")


def log(msg):
    """写入日志"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def start_service():
    """启动收集服务"""
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text())
        try:
            os.kill(pid, 0)
            print(f"Service already running (PID: {pid})")
            return
        except ProcessLookupError:
            pass
    
    # 创建数据文件
    DATA_FILE.touch(exist_ok=True)
    LOG_FILE.touch(exist_ok=True)
    
    # 守护进程化
    pid = os.fork()
    if pid > 0:
        # 父进程等待确认
        time.sleep(1)
        if PID_FILE.exists():
            actual_pid = int(PID_FILE.read_text())
            print(f"Service started (PID: {actual_pid})")
            print(f"Log: {LOG_FILE}")
        return
    
    # 子进程
    os.setsid()
    
    # 写入 PID
    PID_FILE.write_text(str(os.getpid()))
    
    log("=" * 50)
    log("Service starting...")
    
    # 初始化监控器
    try:
        monitor = get_monitor()
        log("Monitor initialized")
    except Exception as e:
        log(f"Monitor init failed: {e}")
        return
    
    # 文件监听线程
    def watch_file():
        last_size = DATA_FILE.stat().st_size
        log(f"Watching {DATA_FILE} (starting at {last_size} bytes)")
        
        while True:
            time.sleep(0.5)
            try:
                current_size = DATA_FILE.stat().st_size
                if current_size > last_size:
                    with open(DATA_FILE, 'r') as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                    
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            process_line(line)
                    
                    last_size = current_size
            except Exception as e:
                log(f"[Watch] Error: {e}")
    
    thread = threading.Thread(target=watch_file, daemon=True)
    thread.start()
    log("File watcher started")
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Interrupted")
    finally:
        log("Service stopping...")
        PID_FILE.unlink(missing_ok=True)


def process_line(line: str):
    """处理一行数据"""
    try:
        import base64
        data = json.loads(line)
        
        # 检查是否是 base64 编码
        if data.get("encoded"):
            user = base64.b64decode(data["user"]).decode('utf-8')
            assistant = base64.b64decode(data["assistant"]).decode('utf-8')
        else:
            user = data.get("user", "")
            assistant = data.get("assistant", "")
        
        if user and assistant:
            ts = datetime.now(timezone.utc)
            user_id = f"svc_u_{int(ts.timestamp() * 1000)}"
            assist_id = f"svc_a_{int(ts.timestamp() * 1000)}"
            
            process_message("user", user, timestamp=ts, message_id=user_id)
            process_message("assistant", assistant, timestamp=ts, message_id=assist_id)
            
            log(f"Collected: {user[:40]}... -> {assistant[:40]}...")
        else:
            log(f"Skipped (empty): {line[:50]}")
    except json.JSONDecodeError:
        log(f"Invalid JSON: {line[:50]}")
    except Exception as e:
        log(f"Error processing: {e}")


def stop_service():
    """停止服务"""
    if not PID_FILE.exists():
        print("Service not running")
        return
    
    pid = int(PID_FILE.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        # 等待进程结束
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except ProcessLookupError:
                break
        print(f"Service stopped (PID: {pid})")
    except ProcessLookupError:
        print("Service not running")
    finally:
        PID_FILE.unlink(missing_ok=True)


def send_data(user: str, assistant: str):
    """发送数据到服务（支持多行）"""
    if not PID_FILE.exists():
        print("Error: Service not running. Start with: python3 collect_service.py start")
        return False
    
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)  # 检查进程是否存在
    except (ProcessLookupError, ValueError):
        print("Error: Service not running (stale PID file)")
        PID_FILE.unlink(missing_ok=True)
        return False
    
    # 使用 base64 编码避免特殊字符问题
    import base64
    data = {
        "user": base64.b64encode(user.encode('utf-8')).decode('ascii'),
        "assistant": base64.b64encode(assistant.encode('utf-8')).decode('ascii'),
        "encoded": True
    }
    
    with open(DATA_FILE, 'a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print("✓ Data sent to service")
    return True


def send_data_interactive():
    """交互式发送数据（支持多行）"""
    if not PID_FILE.exists():
        print("Error: Service not running")
        return False
    
    print("Enter user message (Ctrl+D or empty line to finish):")
    user_lines = []
    try:
        while True:
            line = input()
            if line == "":
                break
            user_lines.append(line)
    except EOFError:
        pass
    
    if not user_lines:
        print("No user message, cancelled")
        return False
    
    print("\nEnter assistant reply (Ctrl+D or empty line to finish):")
    assistant_lines = []
    try:
        while True:
            line = input()
            if line == "":
                break
            assistant_lines.append(line)
    except EOFError:
        pass
    
    user = "\n".join(user_lines)
    assistant = "\n".join(assistant_lines)
    
    return send_data(user, assistant)


def show_status():
    """显示状态"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text())
            os.kill(pid, 0)
            print(f"Service running (PID: {pid})")
            print(f"Data file: {DATA_FILE}")
            print(f"Log file: {LOG_FILE}")
            if DATA_FILE.exists():
                print(f"Data file size: {DATA_FILE.stat().st_size} bytes")
            if LOG_FILE.exists():
                print(f"Log file size: {LOG_FILE.stat().st_size} bytes")
            return
        except (ProcessLookupError, ValueError):
            print("Service not running (stale PID file)")
            PID_FILE.unlink(missing_ok=True)
    
    print("Service not running")


def show_logs(lines: int = 20):
    """显示日志"""
    if not LOG_FILE.exists():
        print("No log file")
        return
    
    with open(LOG_FILE, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line.rstrip())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 collect_service.py start       # 启动服务")
        print("  python3 collect_service.py stop        # 停止服务")
        print("  python3 collect_service.py status      # 查看状态")
        print("  python3 collect_service.py log [N]     # 查看日志 (默认20行)")
        print("  python3 collect_service.py send 'user' 'assistant'  # 发送数据（单行）")
        print("  python3 collect_service.py send-interactive          # 发送数据（多行交互式）")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "start":
        start_service()
    elif cmd == "stop":
        stop_service()
    elif cmd == "status":
        show_status()
    elif cmd == "log":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        show_logs(lines)
    elif cmd == "send":
        if len(sys.argv) >= 4:
            # 命令行模式（单行）
            send_data(sys.argv[2], sys.argv[3])
        else:
            # 交互式模式（多行）
            send_data_interactive()
    elif cmd == "send-interactive":
        # 强制交互式模式
        send_data_interactive()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
