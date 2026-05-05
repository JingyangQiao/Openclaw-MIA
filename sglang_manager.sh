#!/bin/bash
# SGLang 服务管理脚本 - 支持热更新模型权重

set -e

# 配置
SGLANG_PORT=${SGLANG_PORT:-30000}
SGLANG_MODEL=${SGLANG_MODEL:-"Qwen/Qwen3-8B"}
SGLANG_TP_SIZE=${SGLANG_TP_SIZE:-2}
SGLANG_EXTRA_ARGS=${SGLANG_EXTRA_ARGS:-""}

# 工作目录
WORK_DIR=""
PID_FILE="./tmp/sglang_server.pid"
LOG_FILE="./tmp/sglang_server.log"
CURRENT_MODEL_FILE="./tmp/sglang_current_model.txt"

# 权重目录
WEIGHT_DIR="${WORK_DIR}/weights"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_latest_weight() {
    # 获取最新的权重文件
    if [ -d "$WEIGHT_DIR" ]; then
        # 查找 tar.gz 文件
        latest=$(ls -t "$WEIGHT_DIR"/*.tar.gz 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            echo "$latest"
            return 0
        fi
    fi
    return 1
}

extract_weight() {
    local tar_file="$1"
    local extract_dir="${WORK_DIR}/extracted_models/$(basename "$tar_file" .tar.gz)"
    
    log "Extracting $tar_file to $extract_dir"
    
    # 如果已存在，先删除
    if [ -d "$extract_dir" ]; then
        rm -rf "$extract_dir"
    fi
    
    mkdir -p "$extract_dir"
    tar -xzf "$tar_file" -C "$extract_dir" --strip-components=1
    
    # 查找模型目录（可能在子目录中）
    model_dir=$(find "$extract_dir" -name "config.json" -type f | head -1 | xargs dirname)
    if [ -n "$model_dir" ]; then
        echo "$model_dir"
    else
        echo "$extract_dir"
    fi
}

start_sglang() {
    local model_path="$1"
    
    log "Starting SGLang with model: $model_path"
    
    # 如果服务已在运行，先停止
    if [ -f "$PID_FILE" ]; then
        old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log "Stopping existing SGLang server (PID: $old_pid)"
            kill "$old_pid" 2>/dev/null || true
            sleep 5
            pkill -9 -f "sglang" 2>/dev/null || true
            sleep 2
        fi
    fi
    
    # 启动新服务
    log "Starting SGLang server on port $SGLANG_PORT"
    
    # 使用 nohup 后台运行
    nohup python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --port "$SGLANG_PORT" \
        --tp "$SGLANG_TP_SIZE" \
        $SGLANG_EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    echo "$model_path" > "$CURRENT_MODEL_FILE"
    
    log "SGLang server started with PID: $new_pid"
    
    # 等待服务就绪
    log "Waiting for SGLang to be ready..."
    for i in {1..60}; do
        if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
            log "SGLang is ready!"
            return 0
        fi
        sleep 2
    done
    
    log "ERROR: SGLang failed to start within 120 seconds"
    return 1
}

stop_sglang() {
    log "Stopping SGLang server"
    
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 5
        fi
        rm -f "$PID_FILE"
    fi
    
    pkill -9 -f "sglang" 2>/dev/null || true
    
    log "SGLang server stopped"
}

# 热更新权重（不重启服务）
hot_update() {
    log "Checking for new weights for hot update..."
    
    latest_weight=$(get_latest_weight)
    if [ -z "$latest_weight" ]; then
        log "No weights found in $WEIGHT_DIR"
        return 1
    fi
    
    log "Found latest weight: $latest_weight"
    
    # 检查是否已经是当前模型
    if [ -f "$CURRENT_MODEL_FILE" ]; then
        current_model=$(cat "$CURRENT_MODEL_FILE")
        if [ "$current_model" = "$latest_weight" ]; then
            log "Already using the latest model"
            return 0
        fi
    fi
    
    # 检查 SGLang 是否正在运行
    if ! curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        log "SGLang is not running, starting with new model..."
        model_dir=$(extract_weight "$latest_weight")
        start_sglang "$model_dir"
        return $?
    fi
    
    # 解压权重
    model_dir=$(extract_weight "$latest_weight")
    log "Model extracted to: $model_dir"
    
    # 调用 SGLang 热更新 API
    log "Calling SGLang hot update API..."
    
    response=$(curl -s -X POST "http://localhost:${SGLANG_PORT}/update_weights_from_disk" \
        -H "Content-Type: application/json" \
        -d "{
            \"model_path\": \"$model_dir\",
            \"abort_all_requests\": false,
            \"torch_empty_cache\": true
        }" 2>&1)
    
    log "SGLang response: $response"
    
    # 检查响应
    if echo "$response" | grep -q '"success":true'; then
        echo "$latest_weight" > "$CURRENT_MODEL_FILE"
        log "Hot update successful!"
        
        # 清理旧的解压目录（保留最近3个）
        cleanup_old_extracted_models
        
        return 0
    else
        log "Hot update failed: $response"
        return 1
    fi
}

# 清理旧的解压目录
cleanup_old_extracted_models() {
    local extract_base="${WORK_DIR}/extracted_models"
    if [ -d "$extract_base" ]; then
        # 保留最新的3个目录
        ls -t "$extract_base" | tail -n +4 | while read dir; do
            log "Cleaning up old model: $extract_base/$dir"
            rm -rf "$extract_base/$dir"
        done
    fi
}

status() {
    if curl -s "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        log "SGLang is running on port $SGLANG_PORT"
        if [ -f "$CURRENT_MODEL_FILE" ]; then
            log "Current model: $(cat "$CURRENT_MODEL_FILE")"
        fi
    else
        log "SGLang is not running"
    fi
}

# 主命令
case "$1" in
    start)
        if [ -n "$2" ]; then
            start_sglang "$2"
        else
            # 尝试使用最新权重，否则使用默认模型
            latest_weight=$(get_latest_weight)
            if [ -n "$latest_weight" ]; then
                model_dir=$(extract_weight "$latest_weight")
                start_sglang "$model_dir"
            else
                start_sglang "$SGLANG_MODEL"
            fi
        fi
        ;;
    stop)
        stop_sglang
        ;;
    restart)
        stop_sglang
        sleep 2
        $0 start "$2"
        ;;
    update|hot_update)
        hot_update
        ;;
    status)
        status
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|update|status|logs} [model_path]"
        echo ""
        echo "Commands:"
        echo "  start [model_path]  - Start SGLang server"
        echo "  stop                - Stop SGLang server"
        echo "  restart [model_path]- Restart SGLang server"
        echo "  update              - Hot update to latest trained weight (no restart)"
        echo "  status              - Check SGLang status"
        echo "  logs                - View SGLang logs"
        echo ""
        echo "Environment variables:"
        echo "  SGLANG_PORT         - Server port (default: 30000)"
        echo "  SGLANG_MODEL        - Default model path (default: Qwen/Qwen3-8B)"
        exit 1
        ;;
esac
