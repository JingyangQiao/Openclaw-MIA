#!/usr/bin/env python3
"""
OpenClaw / 飞书 jsonl → 训练侧 conversation_complete 工具（单文件）

飞书落盘文件名建议用 make_feishu_jsonl_filename(session_id, turn)，带随机 hex 后缀，避免同轮次/同秒覆盖。

【inbox】监控目录，凑满一批 .jsonl 后向 Server B 的 /receive POST（json=body）
  python openclaw_forward.py WATCH_DIR [--server-b-url URL] [--interval SEC] [--recursive] [--once] [--state-file PATH]
  环境变量: SERVER_B_URL, OPENCLAW_BATCH_SIZE,
            OPENCLAW_MERGE_SESSION_TURNS 默认 0：一批内每个 jsonl 单独 POST（凑满 N 个文件则 N 次请求，可并行观感）；
            设为 1 时才把同会话多 turn 文件合并成一条 conversation_complete。
            OPENCLAW_PRINT_CONVERTED (=0 关闭) 打印转换后的字典；OPENCLAW_PRINT_MAX_CHARS 单条最大字符数（默认 20000）

【mock】向 Server A 的 /collect 发模拟数据（原 mock_data_sender）
  python openclaw_forward.py mock [ROUNDS]
  python openclaw_forward.py mock continuous [INTERVAL]
  python openclaw_forward.py mock watch WATCH_DIR [POLL_INTERVAL_SEC]
  环境变量: SERVER_A_URL, MOCK_SENDER_PER_ROUND, MOCK_SENDER_FAIL_PROBABILITY, MOCK_SENDER_PAUSE_SEC,
            MOCK_WATCH_STATE_FILE, MOCK_WATCH_POLL_SEC
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# --- Server B（inbox 默认）---
DEFAULT_SERVER_B_URL = ("")

# --- Server A（mock）---
SERVER_A_URL = os.environ.get(
    "SERVER_A_URL",
    "",
)

FAIL_PROBABILITY = float(os.environ.get("MOCK_SENDER_FAIL_PROBABILITY", "0"))
PER_ROUND = int(os.environ.get("MOCK_SENDER_PER_ROUND", "4"))
PAUSE_BETWEEN_ROUNDS_SEC = float(os.environ.get("MOCK_SENDER_PAUSE_SEC", "10"))

SAMPLE_QUESTIONS = [
    "如何学习Python编程？",
    "什么是机器学习？",
    "如何优化深度学习模型？",
    "Transformer架构的原理是什么？",
    "如何使用PyTorch训练神经网络？",
    "什么是注意力机制？",
    "如何进行数据预处理？",
    "模型过拟合了怎么办？",
    "如何选择合适的学习率？",
    "什么是强化学习？",
]

SAMPLE_PLANS = [
    "1. 学习基础概念 2. 实践练习 3. 项目实战",
    "1. 理解原理 2. 查看示例 3. 动手实现",
    "1. 分析问题 2. 查找资料 3. 验证方案",
    "1. 阅读论文 2. 理解代码 3. 复现结果",
    "1. 准备数据 2. 搭建模型 3. 训练调优",
]


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# 例（旧）: 20260418_221124_feishu-2cf35c90022c4845_turn1.jsonl
# 例（推荐，防覆盖）: 20260418_221124_feishu-2cf35c90022c4845_turn1_a3f9c2d1.jsonl  ← 末尾 _4~16 位 hex
_FEISHU_JSONL_NAME = re.compile(
    r"^(?P<ts>\d{8}_\d{6})_(?P<session>feishu-[^_]+)_turn(?P<turn>\d+)(_(?P<uniq>[a-f0-9]{4,16}))?\.jsonl$",
    re.IGNORECASE,
)

FeishuNameMeta = Tuple[str, int, str, str]  # session_id, turn, ts_prefix, uniq_suffix or ""


def make_feishu_jsonl_filename(
    session_id: str,
    turn: int,
    *,
    when: Optional[datetime] = None,
    unique_suffix: Optional[str] = None,
) -> str:
    """
    推荐给导出端写入磁盘用的文件名：时间戳 + 会话 + 轮次 + 随机 hex，避免同会话同轮次同秒覆盖。
    session_id 应用 feishu- 前缀形式（若缺省则原样拼进文件名，由调用方保证合法）。
    """
    from datetime import timezone

    dt = when or datetime.now(timezone.utc)
    ts_prefix = dt.strftime("%Y%m%d_%H%M%S")
    if unique_suffix:
        uq = re.sub(r"[^a-fA-F0-9]", "", unique_suffix)[:16]
    else:
        uq = uuid.uuid4().hex[:8]
    if not uq:
        uq = uuid.uuid4().hex[:8]
    sid = session_id.strip()
    return f"{ts_prefix}_{sid}_turn{int(turn)}_{uq.lower()}.jsonl"


def parse_feishu_jsonl_filename(path: Path) -> Optional[FeishuNameMeta]:
    m = _FEISHU_JSONL_NAME.match(path.name)
    if not m:
        return None
    uniq = (m.group("uniq") or "").lower()
    return (m.group("session"), int(m.group("turn")), m.group("ts"), uniq)


def _jsonl_path_sort_key(path: Path) -> Tuple[Any, ...]:
    meta = parse_feishu_jsonl_filename(path)
    if meta:
        session, turn, ts_prefix, uniq = meta
        return (0, session, turn, ts_prefix, uniq, path.name.lower())
    return (1, path.stat().st_mtime, path.name.lower())


def _feishu_turn_sort_within_group(path: Path) -> Tuple[Any, ...]:
    meta = parse_feishu_jsonl_filename(path)
    if meta:
        _, turn, ts_prefix, uniq = meta
        return (turn, ts_prefix, uniq, path.name.lower())
    return (0, "", "", path.name.lower())


def _messages_to_turns(msgs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "assistant"))
        content = m.get("content", "")
        out.append({"role": role, "content": content})
    return out


def merge_conversation_complete_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(payloads) == 1:
        return copy.deepcopy(payloads[0])
    last = payloads[-1]
    out: Dict[str, Any] = copy.deepcopy(last)
    all_turns: List[Dict[str, Any]] = []
    success = True
    exec_time = 0.0
    source: Optional[str] = None
    for pl in payloads:
        data = pl.get("data")
        if not isinstance(data, dict):
            continue
        turns = data.get("turns")
        if isinstance(turns, list) and turns:
            all_turns.extend(_messages_to_turns(turns))
        success = success and bool(data.get("success", True))
        exec_time += float(data.get("execution_time") or 0.0)
        if source is None and data.get("source") is not None:
            source = str(data.get("source"))
    inner = out.get("data")
    if not isinstance(inner, dict):
        inner = {}
    inner["turns"] = all_turns
    inner["success"] = success
    inner["execution_time"] = exec_time
    if source is not None:
        inner["source"] = source
    if all_turns:
        inner["user_question"] = next(
            (x["content"] for x in all_turns if x["role"] == "user"),
            all_turns[0]["content"],
        )
        inner["plan"] = next(
            (x["content"] for x in reversed(all_turns) if x["role"] == "assistant"),
            all_turns[-1]["content"],
        )
    out["data"] = inner
    out["session_id"] = str(last.get("session_id", ""))
    out["timestamp"] = str(last.get("timestamp") or _utc_now_iso())
    return out


def _group_paths_for_merge(paths: List[Path], merge: bool) -> List[List[Path]]:
    if not merge:
        return [[p] for p in paths]
    buckets: Dict[str, List[Path]] = {}
    order: List[str] = []
    for p in paths:
        meta = parse_feishu_jsonl_filename(p)
        key = meta[0] if meta else str(p.resolve())
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(p)
    out: List[List[Path]] = []
    for key in order:
        grp = buckets[key]
        grp.sort(key=_feishu_turn_sort_within_group)
        out.append(grp)
    return out


def _turn_data_dict(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = obj.get("data")
    if isinstance(data, dict) and isinstance(data.get("messages"), list) and data["messages"]:
        return data
    if isinstance(obj.get("messages"), list) and obj["messages"]:
        return {k: v for k, v in obj.items() if k not in ("type", "timestamp")}
    return None


def convert_openclaw_record(obj: Dict[str, Any], _depth: int = 0) -> Dict[str, Any]:
    if _depth > 4:
        return obj

    t = obj.get("type")

    if t is None and isinstance(obj.get("messages"), list) and obj["messages"]:
        wrapped = {
            "type": "conversation_turn",
            "session_id": str(obj.get("session_id", "")),
            "data": obj,
            "timestamp": str(obj.get("timestamp", "")),
        }
        return convert_openclaw_record(wrapped, _depth + 1)

    if t == "conversation_turn":
        data = _turn_data_dict(obj)
        if not data:
            logging.warning("conversation_turn 无 messages，跳过该行")
            return {}
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return {}
        turns = _messages_to_turns(msgs)
        if not turns:
            return {}
        session_id = str(obj.get("session_id") or data.get("session_id") or "")
        first_user = next((x["content"] for x in turns if x["role"] == "user"), turns[0]["content"])
        last_asst = next(
            (x["content"] for x in reversed(turns) if x["role"] == "assistant"),
            turns[-1]["content"],
        )
        inner: Dict[str, Any] = {
            "user_question": first_user,
            "plan": last_asst,
            "turns": turns,
            "success": bool(data.get("success", True)),
            "execution_time": float(data.get("execution_time") or 0.0),
            "source": str(data.get("source") or "openclaw"),
        }
        return {
            "type": "conversation_complete",
            "session_id": session_id,
            "data": inner,
            "timestamp": str(obj.get("timestamp") or _utc_now_iso()),
        }

    if t == "conversation_complete":
        out = copy.deepcopy(obj)
        out.pop("server", None)
        data = out.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("turns"), list) and data["turns"]:
                data["turns"] = _messages_to_turns(data["turns"])
            elif isinstance(data.get("messages"), list) and data["messages"]:
                data["turns"] = _messages_to_turns(data["messages"])
            data.pop("messages", None)
            for k in ("prompt_text", "response_text", "tool_calls", "next_state"):
                data.pop(k, None)
        return out

    logging.warning("未知 type=%s，跳过该行", t)
    return {}


def _print_converted_payload(path: Path, line_no: int, conv: Dict[str, Any]) -> None:
    """调试：打印 jsonl 某行转换后的字典（stderr，UTF-8）。"""
    v = os.environ.get("OPENCLAW_PRINT_CONVERTED", "1").strip().lower()
    if v in ("0", "false", "no", "off", ""):
        return
    try:
        text = json.dumps(conv, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        text = str(conv)
    max_chars = int(os.environ.get("OPENCLAW_PRINT_MAX_CHARS", "20000"))
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, OPENCLAW_PRINT_MAX_CHARS={max_chars}]"
    print(
        f"[openclaw] --- converted dict --- file={path.name} line={line_no} ---\n{text}\n[openclaw] --- end ---\n",
        file=sys.stderr,
        flush=True,
    )


def payloads_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    name_meta = parse_feishu_jsonl_filename(path)
    text = path.read_text(encoding="utf-8")
    line_no = 0
    for line in text.splitlines():
        line_no += 1
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            logging.warning("跳过非对象行: %s", path.name)
            continue
        conv = convert_openclaw_record(obj)
        if not conv or conv.get("type") != "conversation_complete":
            continue
        if name_meta:
            conv["session_id"] = name_meta[0]  # 与文件名中的 feishu- 会话一致
        _print_converted_payload(path, line_no, conv)
        out.append(conv)
    return out


def load_processed(state_path: Path) -> Set[str]:
    if not state_path.exists():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return set(data.get("processed", []))
    except Exception:
        return set()


def save_processed(state_path: Path, processed: Set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"processed": sorted(processed)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(state_path)


def post_to_server_b(
    session: requests.Session,
    post_url: str,
    payload: Dict[str, Any],
    timeout: float,
    *,
    source_label: str = "",
) -> bool:
    try:
        resp = session.post(post_url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            extra = f" {source_label}" if source_label else ""
            logging.info(
                "POST ok type=%s session_id=%s%s",
                payload.get("type"),
                payload.get("session_id"),
                extra,
            )
            return True
        logging.error("HTTP %s: %s", resp.status_code, resp.text[:500])
        return False
    except requests.RequestException as e:
        logging.error("POST 失败: %s", e)
        return False


def iter_jsonl_files(watch_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = list(watch_dir.rglob("*.jsonl"))
    else:
        files = list(watch_dir.glob("*.jsonl"))
    return sorted(
        (p for p in files if p.is_file() and not p.name.startswith(".")),
        key=_jsonl_path_sort_key,
    )


# --- mock: 模拟发送与 watch → /collect ---


def _post_collect_payload(payload: Dict[str, Any]) -> bool:
    try:
        resp = requests.post(
            f"{SERVER_A_URL}/collect",
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            sid = str(payload.get("session_id", ""))[:8]
            print(f"[mock] watch: posted conversation_complete session_id={sid}...")
            return True
        print(f"[mock] watch: HTTP failed: {resp.status_code}")
        return False
    except Exception as e:
        print(f"[mock] watch: Error: {e}")
        return False


def _first_payload_from_jsonl_file(path: Path) -> Optional[Dict[str, Any]]:
    pls = payloads_from_jsonl(path)
    return pls[0] if pls else None


def watch_jsonl_to_collect(watch_dir: Path, poll_interval_sec: float) -> None:
    watch_dir = watch_dir.expanduser().resolve()
    if not watch_dir.is_dir():
        print(f"[mock] watch: 不是目录: {watch_dir}")
        return

    state_path = Path(
        os.environ.get("MOCK_WATCH_STATE_FILE", str(watch_dir / ".mock_sender_processed.json"))
    ).expanduser().resolve()

    need = max(1, PER_ROUND)
    preview = SERVER_A_URL[:48] + "..." if len(SERVER_A_URL) > 48 else SERVER_A_URL
    print(
        f"[mock] watch: dir={watch_dir} need_new_files={need} poll={poll_interval_sec}s "
        f"state={state_path} SERVER_A_URL={preview}"
    )

    processed = load_processed(state_path)

    try:
        while True:
            files = sorted(
                (p for p in watch_dir.glob("*.jsonl") if p.is_file() and not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
            )
            pending = [p for p in files if str(p.resolve()) not in processed]
            if len(pending) < need:
                time.sleep(poll_interval_sec)
                continue

            batch = pending[:need]
            print(f"\n[mock] watch: batch {need} files: {[p.name for p in batch]}")

            payloads: List[Dict[str, Any]] = []
            for p in batch:
                try:
                    pl = _first_payload_from_jsonl_file(p)
                except Exception as e:
                    print(f"[mock] watch: 读取失败 {p.name}: {e}")
                    pl = None
                if not pl:
                    print(f"[mock] watch: 无有效 conversation_complete 行，本批放弃: {p.name}")
                    payloads = []
                    break
                payloads.append(pl)

            if len(payloads) != need:
                time.sleep(poll_interval_sec)
                continue

            all_ok = True
            for pl in payloads:
                if not _post_collect_payload(pl):
                    all_ok = False
                    break

            if all_ok:
                for p in batch:
                    processed.add(str(p.resolve()))
                save_processed(state_path, processed)
                print(f"[mock] watch: 本批 {need} 个文件已标记完成")
            else:
                print("[mock] watch: 本批发送失败，不标记已处理，稍后重试")

            time.sleep(poll_interval_sec)
    except KeyboardInterrupt:
        print("\n[mock] watch: 已停止")


def send_one_message() -> bool:
    if FAIL_PROBABILITY > 0 and random.random() < FAIL_PROBABILITY:
        print(f"[mock] Simulated send failure ({FAIL_PROBABILITY:.0%}), skip HTTP")
        return False

    session_id = str(uuid.uuid4())
    question = random.choice(SAMPLE_QUESTIONS)
    plan = random.choice(SAMPLE_PLANS)
    success = True

    data = {
        "type": "conversation_complete",
        "session_id": session_id,
        "data": {
            "user_question": question,
            "plan": plan,
            "workflow": [
                {"step": 1, "action": "analyze", "result": "问题分析完成"},
                {"step": 2, "action": "search", "result": "找到相关资料"},
                {"step": 3, "action": "generate", "result": "生成回答"},
                {"step": 4, "action": "judge", "result": "检查回答对错"},
            ],
            "success": success,
            "execution_time": random.uniform(5.0, 15.0),
            "source": "mock_sender",
        },
        "timestamp": datetime.now().isoformat(),
    }

    try:
        resp = requests.post(
            f"{SERVER_A_URL}/collect",
            json=data,
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"[mock] Sent 1 message: {session_id[:8]}...")
            return True
        print(f"[mock] HTTP failed: {resp.status_code}")
        return False
    except Exception as e:
        print(f"[mock] Error: {e}")
        return False


def send_round() -> int:
    ok = 0
    for _ in range(PER_ROUND):
        if send_one_message():
            ok += 1
    return ok


def send_conversation_batch(num_rounds: int = 1) -> int:
    print(
        f"[mock] {num_rounds} round(s), {PER_ROUND} msgs/round, "
        f"fail_prob={FAIL_PROBABILITY}, pause_between_rounds={PAUSE_BETWEEN_ROUNDS_SEC}s"
    )
    total_ok = 0
    for r in range(num_rounds):
        print(f"\n[mock] --- Round {r + 1}/{num_rounds} ---")
        total_ok += send_round()
        if r < num_rounds - 1:
            time.sleep(PAUSE_BETWEEN_ROUNDS_SEC)
    print(f"[mock] Completed: {total_ok}/{num_rounds * PER_ROUND} reached server OK")
    return total_ok


def continuous_send(interval: Optional[float] = None) -> None:
    pause = PAUSE_BETWEEN_ROUNDS_SEC if interval is None else interval
    print(
        f"[mock] Continuous: {PER_ROUND} msgs/round, then wait {pause}s, fail_prob={FAIL_PROBABILITY}"
    )

    total_ok = 0
    tick = 0

    try:
        while True:
            tick += 1
            print(f"\n[mock] ===== Tick #{tick} =====")
            total_ok += send_round()
            print(f"[mock] Total OK messages: {total_ok}")
            print(f"[mock] Waiting {pause}s...")
            time.sleep(pause)
    except KeyboardInterrupt:
        print(f"\n[mock] Stopped. Ticks: {tick}, total OK msgs: {total_ok}")


def _main_mock_cli() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("watch", "--watch", "-w"):
        if len(sys.argv) < 3:
            print(
                "用法: python openclaw_inbox_forwarder.py mock watch WATCH_DIR [POLL_INTERVAL_SEC]",
                file=sys.stderr,
            )
            sys.exit(2)
        watch_path = Path(sys.argv[2])
        poll = float(sys.argv[3]) if len(sys.argv) > 3 else float(os.environ.get("MOCK_WATCH_POLL_SEC", "5"))
        watch_jsonl_to_collect(watch_path, poll)
    elif len(sys.argv) > 1 and sys.argv[1] in ("continuous", "--continuous", "-c"):
        interval = float(sys.argv[2]) if len(sys.argv) > 2 else None
        continuous_send(interval=interval)
    else:
        rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        send_conversation_batch(rounds)


def main_inbox() -> None:
    parser = argparse.ArgumentParser(
        description="监控目录 .jsonl，转换后凑满 N 个文件向 Server B /receive POST"
    )
    parser.add_argument("watch_dir", type=Path, help="监控目录")
    parser.add_argument(
        "--server-b-url",
        type=str,
        default=os.environ.get("SERVER_B_URL", DEFAULT_SERVER_B_URL),
        help="Server B 根 URL（不含 /receive）",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="轮询间隔（秒）")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次 POST 超时（秒）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描")
    parser.add_argument("--once", action="store_true", help="处理完当前可凑批次后退出")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="已处理文件列表（默认 WATCH_DIR/.inbox_forwarder_state.json）",
    )
    args = parser.parse_args()

    watch_dir = args.watch_dir.expanduser().resolve()
    state_path = (
        args.state_file.expanduser().resolve()
        if args.state_file
        else watch_dir / ".inbox_forwarder_state.json"
    )
    batch_size = max(1, int(os.environ.get("OPENCLAW_BATCH_SIZE", "4")))
    merge_session_turns = os.environ.get("OPENCLAW_MERGE_SESSION_TURNS", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )

    base = (args.server_b_url or "").strip().rstrip("/")
    if not base:
        print("必须提供 SERVER_B_URL 或 --server-b-url", file=sys.stderr)
        sys.exit(1)
    post_url = f"{base}/receive"

    if not watch_dir.is_dir():
        print(f"WATCH_DIR 不是目录: {watch_dir}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(
        "watch=%s POST=%s batch_files=%s merge_session_turns=%s",
        watch_dir,
        post_url,
        batch_size,
        merge_session_turns,
    )

    processed = load_processed(state_path)
    http = requests.Session()

    def send_batch(batch: List[Path]) -> bool:
        groups = _group_paths_for_merge(batch, merge_session_turns)
        for group in groups:
            flat: List[Dict[str, Any]] = []
            labels: List[str] = []
            for src in group:
                payloads = payloads_from_jsonl(src)
                if not payloads:
                    logging.error("文件无有效 conversation_complete 行，本批失败: %s", src)
                    return False
                n = len(payloads)
                for i, pl in enumerate(payloads, start=1):
                    flat.append(pl)
                    lab = f"file={src.name}"
                    if n > 1:
                        lab += f" line={i}/{n}"
                    labels.append(lab)

            if merge_session_turns and len(flat) > 1:
                merged = merge_conversation_complete_payloads(flat)
                merge_label = f"merged files={'+'.join(p.name for p in group)}"
                if not post_to_server_b(http, post_url, merged, args.timeout, source_label=merge_label):
                    logging.error("发送失败，本批中止且不标记已处理: %s", group[0])
                    return False
            else:
                for pl, label in zip(flat, labels):
                    if not post_to_server_b(http, post_url, pl, args.timeout, source_label=label):
                        logging.error("发送失败，本批中止且不标记已处理: %s", group[0])
                        return False
        return True

    def run_batches_until_short() -> None:
        while True:
            candidates = iter_jsonl_files(watch_dir, args.recursive)
            pending = [p for p in candidates if str(p.resolve()) not in processed]
            if len(pending) < batch_size:
                return
            batch = pending[:batch_size]
            if send_batch(batch):
                for src in batch:
                    processed.add(str(src.resolve()))
                save_processed(state_path, processed)
                logging.info("本批 %s 个文件已发送并标记完成", batch_size)
            else:
                return

    if args.once:
        run_batches_until_short()
        left = [p for p in iter_jsonl_files(watch_dir, args.recursive) if str(p.resolve()) not in processed]
        if left:
            logging.info("--once 结束: 剩余 %s 个文件不足一批 (%s)", len(left), batch_size)
        return

    while True:
        try:
            run_batches_until_short()
        except Exception:
            logging.exception("scan error")
        time.sleep(args.interval)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        _main_mock_cli()
    else:
        main_inbox()
