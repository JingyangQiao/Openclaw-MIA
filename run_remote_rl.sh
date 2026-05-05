#!/bin/bash
# 远程数据训练脚本 V2 - 集成实时权重转换
# 每次保存检查点后自动转换为HF格式并热更新到SGLang

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"

# 本轮训练所有检查点转换相关日志、脚本、状态文件统一目录（默认相对本脚本所在目录）
LOG_RUN_DIR="${SCRIPT_DIR}/${REMOTE_RL_LOG_DIR:-remote_rl_run_logs}"
CHECKPOINT_MONITOR_PID_FILE="${SCRIPT_DIR}/.remote_rl_checkpoint_monitor.pid"

# 必须先停掉上一轮遗留的 checkpoint 监控再删目录，否则旧进程仍引用已删除的 convert_hook_*.sh
stop_checkpoint_monitor() {
    if [ -f "$CHECKPOINT_MONITOR_PID_FILE" ]; then
        local old_pid
        old_pid=$(cat "$CHECKPOINT_MONITOR_PID_FILE" 2>/dev/null || echo "")
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[Main] Stopping previous checkpoint monitor PID $old_pid"
            kill "$old_pid" 2>/dev/null || true
            sleep 1
            kill -9 "$old_pid" 2>/dev/null || true
        fi
        rm -f "$CHECKPOINT_MONITOR_PID_FILE"
    fi
    # 兜底：命令行里仍挂着本目录下 checkpoint_monitor_*.sh 的 bash（例如无 PID 文件的旧实例）
    if [ -d "$LOG_RUN_DIR" ]; then
        pkill -f "${LOG_RUN_DIR}/checkpoint_monitor_" 2>/dev/null || true
        sleep 1
        pkill -9 -f "${LOG_RUN_DIR}/checkpoint_monitor_" 2>/dev/null || true
    fi
}

stop_checkpoint_monitor

if [ -d "$LOG_RUN_DIR" ]; then
    echo "[Main] Removing existing log directory: $LOG_RUN_DIR"
    rm -rf "$LOG_RUN_DIR"
fi
mkdir -p "$LOG_RUN_DIR"
touch "${LOG_RUN_DIR}/convert.log"
echo "[Main] Log directory (fresh): $LOG_RUN_DIR"
echo "[Main]   - converted_checkpoints_v2.txt, convert_hook.log, checkpoint_monitor.log, convert.log"
echo "[Main]   - convert_hook_*.sh, checkpoint_monitor_*.sh"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
PRM_GPUS=${PRM_GPUS:-2}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

# 模型路径
HF_CKPT=${HF_CKPT:-./Qwen3-8b}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-OpenClaw-RL/openclaw-rl/ckpt/qwen3-4b-remote-rl}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-./Qwen3-8b}

# HF格式输出目录
HF_OUTPUT_BASE="${SAVE_CKPT}_hf"
ORIGIN_HF_DIR="${HF_CKPT}"

# 创建输出目录
mkdir -p "$SAVE_CKPT"
mkdir -p "$HF_OUTPUT_BASE"

# 远程服务器配置
export SERVER_A_URL=""
export SERVER_B_URL=""

# SGLang配置
export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export TP="2"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export PUSH_WEIGHT_TO_A="1"

# ========== 实时转换钩子 ==========
# 创建转换钩子脚本
CONVERT_HOOK_SCRIPT="${LOG_RUN_DIR}/convert_hook_$(date +%Y%m%d_%H%M%S).sh"
cat > "$CONVERT_HOOK_SCRIPT" << 'EOFHOOK'
#!/bin/bash
# 转换钩子 - 在每次保存检查点后调用

ITER_DIR="$1"
HF_OUTPUT_BASE="$2"
ORIGIN_HF_DIR="$3"
SCRIPT_DIR="$4"
SGLANG_PORT="${SGLANG_PORT:-30000}"

if [ -z "$ITER_DIR" ] || [ ! -d "$ITER_DIR" ]; then
    echo "[ConvertHook] Invalid checkpoint directory: $ITER_DIR"
    exit 1
fi

ITER_NAME=$(basename "$ITER_DIR")
HF_OUTPUT_DIR="${HF_OUTPUT_BASE}/${ITER_NAME}"

# 检查是否已转换
if [ -d "$HF_OUTPUT_DIR" ] && [ -f "$HF_OUTPUT_DIR/config.json" ]; then
    echo "[ConvertHook] $ITER_NAME already converted, skipping"
    exit 0
fi

echo "[ConvertHook] =========================================="
echo "[ConvertHook] Converting $ITER_NAME"
echo "[ConvertHook] DCP: $ITER_DIR"
echo "[ConvertHook] HF:  $HF_OUTPUT_DIR"
echo "[ConvertHook] =========================================="

# 运行转换（使用不删除DCP的版本）
cd "$SCRIPT_DIR"
if bash convert_and_update.sh "$ITER_DIR" "$HF_OUTPUT_DIR" "$ORIGIN_HF_DIR" >> __RUN_LOG_DIR__/convert_hook.log 2>&1; then
    echo "[ConvertHook] Conversion successful: $ITER_NAME"
    
    # 触发SGLang热更新
    echo "[ConvertHook] Triggering SGLang hot update..."
    RESPONSE=$(curl -s -X POST "http://localhost:${SGLANG_PORT}/update_weights_from_disk" \
        -H "Content-Type: application/json" \
        -d "{\"model_path\":\"$HF_OUTPUT_DIR\",\"abort_all_requests\":false,\"torch_empty_cache\":true}" 2>&1)
    
    if echo "$RESPONSE" | grep -q '"success":true'; then
        echo "[ConvertHook] Hot update successful"
    else
        echo "[ConvertHook] Hot update failed: $RESPONSE"
    fi
else
    echo "[ConvertHook] Conversion failed: $ITER_NAME"
fi
EOFHOOK
sed -i "s#__RUN_LOG_DIR__#${LOG_RUN_DIR}#g" "$CONVERT_HOOK_SCRIPT"
chmod +x "$CONVERT_HOOK_SCRIPT"

# 创建监控进程 - 使用 inotify 或轮询检测新检查点
MONITOR_SCRIPT="${LOG_RUN_DIR}/checkpoint_monitor_$(date +%Y%m%d_%H%M%S).sh"
cat > "$MONITOR_SCRIPT" << EOFMONITOR
#!/bin/bash
set -o pipefail
SAVE_CKPT="$SAVE_CKPT"
HF_OUTPUT_BASE="$HF_OUTPUT_BASE"
ORIGIN_HF_DIR="$ORIGIN_HF_DIR"
SCRIPT_DIR="$SCRIPT_DIR"
CONVERT_HOOK="$CONVERT_HOOK_SCRIPT"

echo "[CheckpointMonitor] Started at \$(date)"
echo "[CheckpointMonitor] Watching: \$SAVE_CKPT"
echo "[CheckpointMonitor] Hook: \$CONVERT_HOOK"

# 记录已处理的检查点
PROCESSED_FILE="${LOG_RUN_DIR}/converted_checkpoints_v2.txt"
touch "\$PROCESSED_FILE"

while true; do
    sleep 30
    
    if [ ! -d "\$SAVE_CKPT" ]; then
        continue
    fi
    
    # 查找所有检查点 - 使用进程替换避免子shell问题
    while read iter_dir; do
        iter_name=\$(basename "\$iter_dir")
        
        # 跳过已处理的
        if grep -q "^\${iter_name}\$" "\$PROCESSED_FILE" 2>/dev/null; then
            continue
        fi
################################################################        
        # 检查是否完整（DCP 分阶段写入，需等待完全写入）
        # .metadata 是 DCP 保存流程最后生成的文件，等它出现才说明写完了
        if [ ! -f "\$iter_dir/.metadata" ]; then
            continue
        fi

        # 稳定性检查：等 10 秒后再次确认文件数量不再增长
        # 避免还在写入中途就触发转换
        sleep 10
        file_count_now=\$(find "\$iter_dir" -type f | wc -l)
        sleep 10
        file_count_later=\$(find "\$iter_dir" -type f | wc -l)
        if [ "\$file_count_now" != "\$file_count_later" ]; then
            echo "[CheckpointMonitor] \$iter_name still writing (\$file_count_now -> \$file_count_later files), skipping for now" | tee -a "${LOG_RUN_DIR}/checkpoint_monitor.log"
            continue
        fi
#################################################################
        echo "[CheckpointMonitor] New checkpoint detected (stable): \$iter_name (\$file_count_now files)"

        
        if [ ! -f "\$CONVERT_HOOK" ]; then
            echo "[CheckpointMonitor] ERROR: hook script missing (log dir likely recreated): \$CONVERT_HOOK" | tee -a "${LOG_RUN_DIR}/checkpoint_monitor.log"
            continue
        fi
        
        # 调用转换钩子（pipefail：bash 失败则整段判失败）
        if bash "\$CONVERT_HOOK" "\$iter_dir" "\$HF_OUTPUT_BASE" "\$ORIGIN_HF_DIR" "\$SCRIPT_DIR" 2>&1 | tee -a "${LOG_RUN_DIR}/checkpoint_monitor.log"; then
            echo "\$iter_name" >> "\$PROCESSED_FILE"
            echo "[CheckpointMonitor] Converted: \$iter_name" | tee -a "${LOG_RUN_DIR}/checkpoint_monitor.log"
        else
            echo "[CheckpointMonitor] Failed to convert: \$iter_name" | tee -a "${LOG_RUN_DIR}/checkpoint_monitor.log"
        fi
    done < <(find "\$SAVE_CKPT" -maxdepth 1 -name "iter_*" -type d | sort)
done
EOFMONITOR
chmod +x "$MONITOR_SCRIPT"

# 启动监控进程
nohup "$MONITOR_SCRIPT" >> "${LOG_RUN_DIR}/checkpoint_monitor.log" 2>&1 &
MONITOR_PID=$!
echo "$MONITOR_PID" > "$CHECKPOINT_MONITOR_PID_FILE"
echo "[Main] Checkpoint monitor started with PID: $MONITOR_PID"
echo "[Main] Monitor log: tail -f ${LOG_RUN_DIR}/checkpoint_monitor.log"
echo "[Main] Convert hook log: tail -f ${LOG_RUN_DIR}/convert_hook.log"
echo "[Main] Placeholder convert.log: ${LOG_RUN_DIR}/convert.log (tee convert_and_update 输出时可自行追加到此文件)"
# =========================================

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path remote_rollout.generate_rollout_remote
   
   --num-rollout 100000000
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score
   
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine 1
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-4b-remote-rl
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"OpenClaw-RL/openclaw-rl:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SERVER_A_URL\": \"${SERVER_A_URL}\",
    \"SERVER_B_URL\": \"${SERVER_B_URL}\",
    \"PUSH_WEIGHT_TO_A\": \"${PUSH_WEIGHT_TO_A}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 OpenClaw-RL/slime/train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PRM_ARGS[@]}
