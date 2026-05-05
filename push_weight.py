import requests
import sys
from pathlib import Path

# 服务器A的代理URL
SERVER_A_URL = ""

def push_weight(weight_path: str):
    """推送权重文件到服务器A"""
    weight_file = Path(weight_path)
    
    if not weight_file.exists():
        print(f"[B] Error: File not found: {weight_path}")
        return False
    
    print(f"[B] Pushing weight: {weight_file.name} ({weight_file.stat().st_size} bytes)")
    
    try:
        with open(weight_file, 'rb') as f:
            files = {'file': (weight_file.name, f, 'application/octet-stream')}
            resp = requests.post(
                f"{SERVER_A_URL}/upload_weight",
                files=files,
                timeout=60
            )
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"[B] Success: {result}")
            return True
        else:
            print(f"[B] Failed: {resp.status_code} - {resp.text}")
            return False
            
    except Exception as e:
        print(f"[B] Error: {e}")
        return False

def create_test_weight():
    """创建一个测试权重文件"""
    test_file = Path("./test_weight.pt")
    # 写入一些模拟数据
    test_file.write_bytes(b"mock pytorch model data " * 1000)
    print(f"[B] Created test weight: {test_file} ({test_file.stat().st_size} bytes)")
    return test_file

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 推送指定文件
        push_weight(sys.argv[1])
    else:
        # 创建并推送测试文件
        test_file = create_test_weight()
        push_weight(test_file)
        # 清理
        test_file.unlink()