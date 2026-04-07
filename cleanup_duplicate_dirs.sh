#!/bin/bash
#
# 清理重复的 bin 和 build 目录
# 保留最新的，删除旧的
#

set -e

PROJECT_DIR="/home/fred/Documents/GitHub/cycleuser/MoXing"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_step()    { echo -e "${CYAN}==>${NC} $1"; }

# 检查目录中的 llama-server/ollama-runner 文件时间
check_dir_date() {
    local dir="$1"
    local latest_file=""
    local latest_time=0
    
    for pattern in "llama-server" "ollama-runner*" "llama-cli"; do
        for file in "$dir"/$pattern; do
            if [ -f "$file" ]; then
                local mtime=$(stat -c %Y "$file" 2>/dev/null || echo 0)
                if [ "$mtime" -gt "$latest_time" ]; then
                    latest_time=$mtime
                    latest_file="$file"
                fi
            fi
        done
    done
    
    echo "$latest_time"
}

# 分析 bin 目录
analyze_bin_dirs() {
    log_step "分析 bin 目录..."
    
    echo ""
    echo "找到以下 bin 目录:"
    echo "----------------------------------------"
    
    declare -A bin_dirs
    
    # 1. 根目录 bin
    if [ -d "$PROJECT_DIR/bin" ]; then
        local mtime=$(check_dir_date "$PROJECT_DIR/bin")
        bin_dirs["$PROJECT_DIR/bin"]=$mtime
        echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $PROJECT_DIR/bin"
    fi
    
    # 2. moxing/bin
    if [ -d "$PROJECT_DIR/moxing/bin" ]; then
        local mtime=$(check_dir_date "$PROJECT_DIR/moxing/bin")
        bin_dirs["$PROJECT_DIR/moxing/bin"]=$mtime
        echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $PROJECT_DIR/moxing/bin"
        
        # 子目录
        for subdir in "$PROJECT_DIR/moxing/bin"/*; do
            if [ -d "$subdir" ]; then
                local sub_mtime=$(check_dir_date "$subdir")
                bin_dirs["$subdir"]=$sub_mtime
                echo "$(date -d @$sub_mtime '+%Y-%m-%d %H:%M')  $subdir"
            fi
        done
    fi
    
    # 3. old_build 中的 bin
    if [ -d "$PROJECT_DIR/old_build" ]; then
        for dir in "$PROJECT_DIR/old_build"/*/bin; do
            if [ -d "$dir" ]; then
                local mtime=$(check_dir_date "$dir")
                bin_dirs["$dir"]=$mtime
                echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $dir (old_build)"
            fi
        done
    fi
    
    echo ""
    
    # 找出最新的
    local newest_dir=""
    local newest_time=0
    
    for dir in "${!bin_dirs[@]}"; do
        if [ "${bin_dirs[$dir]}" -gt "$newest_time" ]; then
            newest_time=${bin_dirs[$dir]}
            newest_dir="$dir"
        fi
    done
    
    echo -e "${GREEN}最新: $newest_dir ($(date -d @$newest_time '+%Y-%m-%d %H:%M'))${NC}"
    echo ""
    
    # 标记要删除的
    echo "建议删除以下旧目录:"
    echo "----------------------------------------"
    for dir in "${!bin_dirs[@]}"; do
        if [ "$dir" != "$newest_dir" ]; then
            echo -e "${RED}DELETE: $dir${NC}"
        fi
    done
}

# 分析 build 目录
analyze_build_dirs() {
    log_step "分析 build 目录..."
    
    echo ""
    echo "找到以下 build 目录:"
    echo "----------------------------------------"
    
    declare -A build_dirs
    
    # 1. 根目录 build
    if [ -d "$PROJECT_DIR/build" ]; then
        for subdir in "$PROJECT_DIR/build"/ollama-runner-*; do
            if [ -d "$subdir" ]; then
                local mtime=$(check_dir_date "$subdir")
                build_dirs["$subdir"]=$mtime
                echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $subdir"
            fi
        done
    fi
    
    # 2. old_build
    if [ -d "$PROJECT_DIR/old_build" ]; then
        for dir in "$PROJECT_DIR/old_build"/build*; do
            if [ -d "$dir" ]; then
                local mtime=$(check_dir_date "$dir")
                build_dirs["$dir"]=$mtime
                echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $dir (old_build)"
            fi
        done
    fi
    
    # 3. ollama 中的 build
    if [ -d "$PROJECT_DIR/ollama" ]; then
        for dir in "$PROJECT_DIR/ollama"/build*; do
            if [ -d "$dir" ]; then
                local mtime=$(check_dir_date "$dir")
                build_dirs["$dir"]=$mtime
                echo "$(date -d @$mtime '+%Y-%m-%d %H:%M')  $dir (ollama)"
            fi
        done
    fi
    
    echo ""
    echo -e "${GREEN}建议保留: $PROJECT_DIR/build/ollama-runner-* (最新构建)${NC}"
    echo ""
    echo "建议删除以下旧目录:"
    echo "----------------------------------------"
    for dir in "${!build_dirs[@]}"; do
        if [[ "$dir" != "$PROJECT_DIR/build/ollama-runner-"* ]]; then
            echo -e "${RED}DELETE: $dir${NC}"
        fi
    done
}

# 执行清理 (需要确认)
execute_cleanup() {
    log_step "执行清理..."
    
    echo ""
    echo "将要删除以下目录:"
    echo "----------------------------------------"
    
    # 收集要删除的目录
    local to_delete=()
    
    # 旧的 bin 目录
    if [ -d "$PROJECT_DIR/bin" ]; then
        to_delete+=("$PROJECT_DIR/bin")
        echo -e "${RED}$PROJECT_DIR/bin${NC}"
    fi
    
    # old_build 中的所有内容
    if [ -d "$PROJECT_DIR/old_build" ]; then
        for dir in "$PROJECT_DIR/old_build"/*; do
            if [ -d "$dir" ]; then
                to_delete+=("$dir")
                echo -e "${RED}$dir${NC}"
            fi
        done
    fi
    
    # ollama 中的旧 build
    if [ -d "$PROJECT_DIR/ollama" ]; then
        for dir in "$PROJECT_DIR/ollama"/build*; do
            if [ -d "$dir" ]; then
                # 保留 vendor/build-* 作为参考
                if [[ "$dir" != *"/vendor/build"* ]]; then
                    to_delete+=("$dir")
                    echo -e "${RED}$dir${NC}"
                fi
            fi
        done
    fi
    
    echo ""
    echo -e "${YELLOW}警告: 此操作不可恢复！${NC}"
    echo -n "确认删除以上目录? (yes/no): "
    read -r confirm
    
    if [ "$confirm" = "yes" ]; then
        echo ""
        log_info "开始删除..."
        for dir in "${to_delete[@]}"; do
            if [ -d "$dir" ]; then
                rm -rf "$dir"
                log_success "删除: $dir"
            fi
        done
        log_success "清理完成！"
    else
        log_info "取消操作"
    fi
}

# 显示最终结构
show_final_structure() {
    echo ""
    log_step "最终目录结构:"
    echo ""
    echo "MoXing/"
    echo "├── build/"
    echo "│   ├── ollama-runner-cpu/"
    echo "│   ├── ollama-runner-cuda/"
    echo "│   ├── ollama-runner-rocm/"
    echo "│   └── ollama-runner-vulkan/"
    echo "├── moxing/bin/"
    echo "│   ├── linux-x64-cpu/"
    echo "│   ├── linux-x64-cuda/"
    echo "│   ├── linux-x64-rocm/"
    echo "│   ├── linux-x64-vulkan/"
    echo "│   ├── ollama-linux-x64-cpu/"
    echo "│   ├── ollama-linux-x64-cuda/"
    echo "│   ├── ollama-linux-x64-rocm/"
    echo "│   └── ollama-linux-x64-vulkan/"
    echo "└── ollama/"
    echo "    └── llama/vendor/"
    echo "        ├── build-rocm/"
    echo "        ├── build-cuda/"
    echo "        └── build-vulkan/"
}

# 主流程
main() {
    echo "=========================================="
    echo "  清理重复目录"
    echo "=========================================="
    echo ""
    
    local command="${1:-analyze}"
    
    case $command in
        analyze|check)
            analyze_bin_dirs
            echo ""
            analyze_build_dirs
            echo ""
            show_final_structure
            ;;
        clean|cleanup)
            analyze_bin_dirs
            echo ""
            analyze_build_dirs
            echo ""
            execute_cleanup
            echo ""
            show_final_structure
            ;;
        *)
            echo "用法:"
            echo "  $0 analyze    # 分析重复目录（默认）"
            echo "  $0 clean      # 执行清理"
            echo ""
            echo "注意: clean 操作需要手动确认"
            ;;
    esac
}

main "$@"
