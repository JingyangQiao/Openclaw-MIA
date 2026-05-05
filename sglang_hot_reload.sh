#!/bin/bash
# =============================================================================
# SGLang Qwen3 模型热更新脚本 (无 jq 依赖版本)
# 功能: 监控指定目录下的HF格式权重，检测到新权重后热更新到SGLang服务
# 运行位置: 服务器A
# =============================================================================

# 不要在循环中因命令失败而退出
set -uo pipefail

# =============================================================================
# 配置参数
# =============================================================================
WATCH_DIR="${WATCH_DIR:-/path/to/hf_weights}"
SGLANG_HOST="${SGLANG_HOST:-localhost}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_API_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
LOG_FILE="${LOG_FILE:-./ckpt-monitor/sglang_hot_reload.log}"
PID_FILE="${PID_FILE:-./ckpt-monitor/sglang_hot_reload.pid}"
STATE_FILE="${STATE_FILE:-./ckpt-monitor/sglang_hot_reload/state.txt}"

# =============================================================================
# 日志函数
# =============================================================================
log() {
    local level="$1"
    shift
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${timestamp} [${level}] $*" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# =============================================================================
# 初始化
# =============================================================================
init() {
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$STATE_FILE")"
    
    if [[ ! -d "$WATCH_DIR" ]]; then
        log_error "监控目录不存在: $WATCH_DIR"
        exit 1
    fi
    
    if [[ ! -f "$STATE_FILE" ]]; then
        touch "$STATE_FILE"
    fi
    
    log_info "初始化完成"
    log_info "监控目录: $WATCH_DIR"
    log_info "SGLang API: $SGLANG_API_URL"
}

# =============================================================================
# 检查 SGLang 服务状态
# =============================================================================
check_sglang_health() {
    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        "${SGLANG_API_URL}/health" 2>/dev/null || echo "000")
    [[ "$response" == "200" ]]
}

# =============================================================================
# 验证是否为完整的HF权重目录
# =============================================================================
is_valid_hf_weights() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] || return 1
    if ls "$dir"/*.bin 1>/dev/null 2>&1 || ls "$dir"/*.safetensors 1>/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# =============================================================================
# 获取权重目录的指纹
# =============================================================================
get_weight_fingerprint() {
    local dir="$1"
    local config_stat=$(stat -c "%Y_%s" "$dir/config.json" 2>/dev/null || echo "")
    local weight_count=$(find "$dir" -name "*.bin" -o -name "*.safetensors" 2>/dev/null | wc -l)
    echo "${config_stat}_${weight_count}"
}

# =============================================================================
# 检查权重是否已处理过
# =============================================================================
is_weight_processed() {
    local fingerprint="$1"
    grep -q "^${fingerprint}|" "$STATE_FILE" 2>/dev/null
}

# =============================================================================
# 标记权重为已处理
# =============================================================================
mark_weight_processed() {
    local fingerprint="$1"
    local weight_path="$2"
    local timestamp=$(date -Iseconds)
    echo "${fingerprint}|${weight_path}|${timestamp}" >> "$STATE_FILE"
}

# =============================================================================
# 热更新 SGLang 模型
# =============================================================================
hot_reload_model() {
    local new_weight_path="$1"
    
    log_info "执行热更新..."
    log_info "权重路径: $new_weight_path"
    
    # 构造 JSON payload
    local payload="{\"model_path\": \"$new_weight_path\", \"abort_all_requests\": false, \"torch_empty_cache\": true}"
    
    local response
    local http_code
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "${SGLANG_API_URL}/update_weights_from_disk" 2>/dev/null || echo -e "\n000")
    
    http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')
    
    if [[ "$http_code" == "200" ]]; then
        log_success "热更新成功"
        log_info "响应: $body"
        return 0
    else
        log_error "热更新失败: HTTP $http_code"
        log_error "响应: $body"
        return 1
    fi
}

# =============================================================================
# 执行热更新流程
# =============================================================================
perform_hot_reload() {
    local weight_path="$1"
    local fingerprint="$2"
    
    log_info "=========================================="
    log_info "检测到新权重: $weight_path"
    log_info "=========================================="
    
    if ! check_sglang_health; then
        log_error "SGLang服务不健康，跳过本次更新"
        return 1
    fi
    
    if hot_reload_model "$weight_path"; then
        mark_weight_processed "$fingerprint" "$weight_path"
        log_success "热更新完成: $weight_path"
        return 0
    else
        log_error "热更新失败"
        return 1
    fi
}

# =============================================================================
# 扫描新权重
# =============================================================================
scan_for_new_weights() {
    log_info "扫描目录: $WATCH_DIR"
    
    local found_new=false
    
    for dir in "$WATCH_DIR"/*; do
        [[ -d "$dir" ]] || continue
        
        if ! is_valid_hf_weights "$dir"; then
            continue
        fi
        
        local fingerprint=$(get_weight_fingerprint "$dir")
        
        if is_weight_processed "$fingerprint"; then
            continue
        fi
        
        log_info "发现新权重: $dir"
        found_new=true
        perform_hot_reload "$dir" "$fingerprint"
    done
    
    [[ "$found_new" == false ]] && log_info "未发现新权重"
}

# =============================================================================
# 信号处理
# =============================================================================
cleanup() {
    log_info "正在停止监控..."
    rm -f "$PID_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# =============================================================================
# 主循环
# =============================================================================
main_loop() {
    log_info "=========================================="
    log_info "SGLang 热更新监控启动"
    log_info "=========================================="
    
    while true; do
        if ! check_sglang_health; then
            log_warn "SGLang服务未响应，等待恢复..."
            sleep "$CHECK_INTERVAL"
            continue
        fi
        
        scan_for_new_weights
        
        log_info "${CHECK_INTERVAL}秒后再次检查..."
        sleep "$CHECK_INTERVAL"
    done
}

# =============================================================================
# 命令处理
# =============================================================================
usage() {
    cat << EOF
用法: $0 [选项] [命令]

命令:
    start       启动监控 (默认)
    stop        停止监控
    status      查看状态
    once        执行一次扫描后退出
    test-api    测试SGLang API

选项:
    -d, --watch-dir DIR     监控目录
    -h, --host HOST         SGLang主机 (默认: localhost)
    -p, --port PORT         SGLang端口 (默认: 30000)
    -i, --interval SECONDS  检查间隔 (默认: 30)
    -l, --log FILE          日志文件
    --help                  显示帮助

示例:
    $0 start
    $0 -d /data/weights -p 30000 start
    nohup $0 -d /data/weights start > /dev/null 2>&1 &
EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--watch-dir) WATCH_DIR="$2"; shift 2 ;;
        -h|--host) SGLANG_HOST="$2"; SGLANG_API_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"; shift 2 ;;
        -p|--port) SGLANG_PORT="$2"; SGLANG_API_URL="http://${SGLANG_HOST}:${SGLANG_PORT}"; shift 2 ;;
        -i|--interval) CHECK_INTERVAL="$2"; shift 2 ;;
        -l|--log) LOG_FILE="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        start|stop|status|once|test-api) COMMAND="$1"; shift ;;
        *) echo "未知参数: $1"; usage; exit 1 ;;
    esac
done

COMMAND="${COMMAND:-start}"

# 执行命令
case $COMMAND in
    start)
        init
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "监控进程已在运行 (PID: $(cat "$PID_FILE"))"
            exit 1
        fi
        echo $$ > "$PID_FILE"
        main_loop
        ;;
    stop)
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" && echo "已停止监控 (PID: $pid)"
            else
                echo "监控进程未运行"
            fi
            rm -f "$PID_FILE"
        else
            echo "未找到PID文件"
        fi
        ;;
    status)
        if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "监控运行中 (PID: $(cat "$PID_FILE"))"
            echo "监控目录: $WATCH_DIR"
            echo "SGLang: $SGLANG_API_URL"
            check_sglang_health && echo "SGLang状态: 正常" || echo "SGLang状态: 异常"
        else
            echo "监控未运行"
        fi
        if [[ -f "$STATE_FILE" ]]; then
            echo "最近处理记录:"
            tail -3 "$STATE_FILE" 2>/dev/null || echo "无记录"
        fi
        ;;
    once)
        init
        scan_for_new_weights
        ;;
    test-api)
        echo "测试SGLang API: $SGLANG_API_URL"
        if check_sglang_health; then
            echo "✓ 服务健康"
            curl -s "${SGLANG_API_URL}/v1/models" 2>/dev/null | head -c 200
            echo ""
        else
            echo "✗ 连接失败"
            exit 1
        fi
        ;;
    *)
        echo "未知命令: $COMMAND"
        usage
        exit 1
        ;;
esac
