#!/bin/bash
# 转换DCP检查点到HuggingFace格式并触发SGLang热更新

set -e

DCP_CHECKPOINT_DIR="$1"
HF_OUTPUT_DIR="$2"
ORIGIN_HF_DIR="$3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [ -z "$DCP_CHECKPOINT_DIR" ] || [ -z "$HF_OUTPUT_DIR" ]; then
    echo "Usage: $0 <dcp_checkpoint_dir> <hf_output_dir> [origin_hf_dir]"
    exit 1
fi

if [ ! -d "$DCP_CHECKPOINT_DIR" ]; then
    log "Error: DCP checkpoint directory does not exist: $DCP_CHECKPOINT_DIR"
    exit 1
fi

ITER_NAME=$(basename "$DCP_CHECKPOINT_DIR")
log "Converting $ITER_NAME to HuggingFace format"

# 如果输出目录已存在且有效，跳过
if [ -d "$HF_OUTPUT_DIR" ] && [ -f "$HF_OUTPUT_DIR/config.json" ] && [ -f "$HF_OUTPUT_DIR/model.safetensors.index.json" ]; then
    log "HF output already exists and valid: $HF_OUTPUT_DIR"
    exit 0
fi

# 如果输出目录存在但不完整，删除重建
if [ -d "$HF_OUTPUT_DIR" ]; then
    log "Removing incomplete HF output: $HF_OUTPUT_DIR"
    rm -rf "$HF_OUTPUT_DIR"
fi

mkdir -p "$HF_OUTPUT_DIR"

export PYTHONPATH="OpenClaw-RL/Megatron-LM:${SLIME_ROOT}:${PYTHONPATH}"
PYTHON="python3"

log "Starting conversion..."
if ! "${PYTHON}" "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "$DCP_CHECKPOINT_DIR" \
    --output-dir "$HF_OUTPUT_DIR" \
    ${ORIGIN_HF_DIR:+--origin-hf-dir "$ORIGIN_HF_DIR"} \
    --force; then
    log "Error: Conversion failed!"
    exit 1
fi

# 验证输出
if [ ! -f "$HF_OUTPUT_DIR/config.json" ] || [ ! -f "$HF_OUTPUT_DIR/model.safetensors.index.json" ]; then
    log "Error: HF output is missing critical files!"
    exit 1
fi

HF_SIZE=$(du -sh "$HF_OUTPUT_DIR" | cut -f1)
log "Conversion completed: $HF_SIZE"

# 删除原始 DCP 文件以节省空间
DCP_SIZE=$(du -sh "$DCP_CHECKPOINT_DIR" | cut -f1)
log "Removing DCP files: $DCP_SIZE"
rm -rf "$DCP_CHECKPOINT_DIR"
log "DCP files removed"

# 触发 SGLang 热更新
SGLANG_PORT=${SGLANG_PORT:-30000}
log "Triggering SGLang hot update..."

RESPONSE=$(curl -s -X POST "http://localhost:${SGLANG_PORT}/update_weights_from_disk" \
    -H "Content-Type: application/json" \
    -d "{\"model_path\":\"$HF_OUTPUT_DIR\",\"abort_all_requests\":false,\"torch_empty_cache\":true}" 2>&1)

if echo "$RESPONSE" | grep -q '"success":true'; then
    log "Hot update successful"
else
    log "Hot update failed: $RESPONSE"
    exit 1
fi
