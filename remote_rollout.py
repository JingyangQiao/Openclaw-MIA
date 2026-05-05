"""
远程 Rollout 生成器
从服务器B的数据缓冲区获取训练数据，替代原有的OpenClaw API方式
"""
import asyncio
import queue
import threading
import time
from typing import List
import requests
import json

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample

# 服务器B的代理URL（从环境变量读取或默认）
import os
SERVER_B_URL = os.environ.get(
    "SERVER_B_URL",
    ""
)

# 数据获取队列
_data_queue = queue.Queue()
_fetcher_thread = None
_fetcher_running = False


def _decode_unicode(obj):
    """递归解码Unicode转义"""
    if isinstance(obj, str):
        try:
            return obj.encode('utf-8').decode('unicode_escape')
        except:
            return obj
    elif isinstance(obj, dict):
        return {k: _decode_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_unicode(item) for item in obj]
    return obj


def _fetch_data_worker(batch_size: int = 4, interval: float = 1.0):
    """后台线程：持续从服务器B获取数据"""
    global _fetcher_running
    _fetcher_running = True
    
    print(f"[RemoteRollout] Data fetcher started, fetching from {SERVER_B_URL}")
    
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while _fetcher_running:
        try:
            # 从服务器B获取数据
            resp = requests.get(
                f"{SERVER_B_URL}/get_data",
                params={"batch_size": batch_size},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                
                for item in items:
                    # 解码Unicode
                    item = _decode_unicode(item)
                    _data_queue.put(item)
                
                # 重置错误计数
                if items:
                    consecutive_errors = 0
                    print(f"[RemoteRollout] Fetched {len(items)} items, queue={_data_queue.qsize()}")
                elif _data_queue.qsize() == 0:
                    # 队列为空且没获取到数据，可能是Server B没数据了
                    if consecutive_errors == 0:
                        print(f"[RemoteRollout] No data available from Server B, queue empty")
            else:
                consecutive_errors += 1
                print(f"[RemoteRollout] Fetch failed: {resp.status_code} (consecutive_errors={consecutive_errors})")
                
        except Exception as e:
            consecutive_errors += 1
            print(f"[RemoteRollout] Fetch error: {e} (consecutive_errors={consecutive_errors})")
        
        # 如果连续错误太多，增加日志级别
        if consecutive_errors >= max_consecutive_errors:
            print(f"[RemoteRollout] WARNING: {consecutive_errors} consecutive errors. "
                  f"Server B may be down or unreachable.")
        
        time.sleep(interval)


def start_data_fetcher(batch_size: int = 4):
    """启动数据获取线程"""
    global _fetcher_thread, _fetcher_running
    
    if _fetcher_thread is None or not _fetcher_thread.is_alive():
        _fetcher_running = True
        _fetcher_thread = threading.Thread(
            target=_fetch_data_worker,
            args=(batch_size,),
            daemon=True
        )
        _fetcher_thread.start()
        print(f"[RemoteRollout] Data fetcher started")


def stop_data_fetcher():
    """停止数据获取线程"""
    global _fetcher_running
    _fetcher_running = False


# 全局tokenizer缓存
_tokenizer = None

def _get_tokenizer():
    """获取或初始化tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            # 使用与训练相同的模型
            model_path = "./Qwen3-8b"
            _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"[RemoteRollout] Tokenizer loaded from {model_path}")
        except Exception as e:
            print(f"[RemoteRollout] Failed to load tokenizer: {e}")
            _tokenizer = None
    return _tokenizer


def _normalize_messages(data: dict, question: str, plan: str) -> List[dict]:
    """将输入数据统一整理为 chat messages。"""
    turns = data.get("turns", [])
    messages: List[dict] = []

    if turns:
        for turn in turns:
            role = turn.get("role", "assistant")
            content = str(turn.get("content", ""))
            if role == "planner":
                role = "assistant"
            if role not in {"system", "user", "assistant"}:
                role = "assistant"
            messages.append({"role": role, "content": content})
    else:
        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(plan)},
        ]

    if not messages:
        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(plan)},
        ]

    return messages


def _conversation_to_sample(conversation: dict) -> Sample:
    """将对话数据转换为SLIME Sample格式"""
    sample = Sample()
    
    # 提取数据
    data = conversation.get("data", conversation)
    
    # 构建prompt和response
    question = data.get("user_question", "")
    plan = data.get("plan", "")
    messages = _normalize_messages(data, question, plan)
    
    # 使用tokenizer进行编码
    tokenizer = _get_tokenizer()
    if tokenizer:
        try:
            if messages[-1]["role"] == "assistant":
                prefix_messages = messages[:-1]
                target_response = messages[-1]["content"]
            else:
                prefix_messages = messages
                target_response = str(plan)

            # 避免前缀为空导致模板行为不稳定
            if not prefix_messages:
                prefix_messages = [{"role": "user", "content": str(question)}]

            full_messages = prefix_messages + [{"role": "assistant", "content": target_response}]

            prompt_text = tokenizer.apply_chat_template(
                prefix_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            response_tokens = full_tokens[len(prompt_tokens) :]

            if not response_tokens:
                # 极端情况下fallback到基础编码，避免空响应破坏训练
                response_tokens = tokenizer.encode(target_response, add_special_tokens=False)
                full_tokens = prompt_tokens + response_tokens

            sample.prompt = prompt_text
            sample.response = target_response
            sample.tokens = full_tokens
            sample.response_length = len(response_tokens)
        except Exception as e:
            print(f"[RemoteRollout] apply_chat_template failed, fallback encode: {e}")
            prompt = messages[0]["content"] if messages else question
            response = messages[-1]["content"] if messages else plan
            sample.prompt = prompt
            sample.response = response
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = tokenizer.encode(response, add_special_tokens=False)
            sample.tokens = prompt_tokens + response_tokens
            sample.response_length = len(response_tokens)
    else:
        # fallback: 使用字符级编码（每个字符一个token ID）
        prompt = messages[0]["content"] if messages else question
        response = messages[-1]["content"] if messages else plan
        sample.prompt = prompt
        sample.response = response
        prompt_tokens = [ord(c) % 100000 for c in prompt]
        response_tokens = [ord(c) % 100000 for c in response]
        sample.tokens = prompt_tokens + response_tokens
        sample.response_length = len(response_tokens)    

    # loss_mask: 只计算response部分的loss
    # 长度应该等于response_length
    sample.loss_mask = [1] * sample.response_length
    sample.rollout_log_probs = [0.0] * sample.response_length
    
    sample.status = Sample.Status.COMPLETED
    sample.index = hash(conversation.get("session_id", str(time.time()))) % 1000000
    sample.group_index = sample.index
    
    # 设置reward
    success = data.get("success", False)
    sample.reward = {"score": 1.0 if success else -1.0}
    
    return sample


async def _drain_remote_data(target_size: int) -> List[List[Sample]]:
    """从队列中获取数据并转换为 Sample；无数据时永久阻塞等待，直至凑满 target_size（需结束请手动停训练）。"""
    data: List[List[Sample]] = []
    start = time.time()
    last_progress = start

    while len(data) < target_size:
        try:
            item = _data_queue.get_nowait()

            if item.get("type") == "conversation_complete":
                sample = _conversation_to_sample(item)
                data.append([sample])

        except queue.Empty:
            if time.time() - last_progress > 10:
                print(
                    f"[RemoteRollout] waiting for data (no timeout): "
                    f"{len(data)}/{target_size}, queue={_data_queue.qsize()}"
                )
                last_progress = time.time()

            await asyncio.sleep(0.1)

    print(f"[RemoteRollout] drained {len(data)} samples in {time.time() - start:.2f}s")

    return data


def generate_rollout_remote(args, rollout_id, data_buffer, evaluation=False):
    """
    替代 generate_rollout_openclaw
    从远程服务器B获取训练数据
    """
    from slime.rollout.sglang_rollout import eval_rollout
    from slime.utils.async_utils import run
    
    if evaluation:
        # 评估模式使用原有逻辑
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output
    
    # 启动数据获取线程
    start_data_fetcher(batch_size=args.rollout_batch_size)
    
    # 等待并获取数据
    completed_samples = run(_drain_remote_data(args.rollout_batch_size))
    
    # 计算指标
    extra_metrics = None
    if completed_samples:
        scores = [s[0].reward.get("score", 0) for s in completed_samples if s]
        if scores:
            avg_score = sum(scores) / len(scores)
            extra_metrics = {"rollout/avg_reward": avg_score}
            print(f"[RemoteRollout] avg_reward={avg_score:.4f} (n={len(scores)})")
    
    return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


# 兼容原有接口
generate_rollout_openclaw_remote = generate_rollout_remote
