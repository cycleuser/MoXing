#!/bin/bash
#
# 端口转发配置脚本
# 支持：ufw, firewalld, iptables
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error()   { echo -e "${RED}[✗]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# 检测防火墙类型
detect_firewall() {
    if command -v ufw &> /dev/null && sudo ufw status &> /dev/null; then
        echo "ufw"
    elif command -v firewall-cmd &> /dev/null; then
        echo "firewalld"
    elif command -v iptables &> /dev/null; then
        echo "iptables"
    else
        echo "none"
    fi
}

# 配置 ufw
setup_ufw() {
    local port=$1
    local ip_range=$2
    
    log_info "配置 ufw..."
    
    # 检查 ufw 状态
    if ! sudo ufw status | grep -q "Status: active"; then
        log_warning "ufw 未启用，正在启用..."
        sudo ufw --force enable
    fi
    
    if [ -n "$ip_range" ]; then
        # 限制特定 IP 范围
        sudo ufw allow from "$ip_range" to any port "$port"
        log_success "允许 $ip_range 访问端口 $port"
    else
        # 允许所有
        sudo ufw allow "$port/tcp"
        log_success "允许所有访问端口 $port"
    fi
    
    sudo ufw status | grep "$port"
}

# 配置 firewalld
setup_firewalld() {
    local port=$1
    local ip_range=$2
    
    log_info "配置 firewalld..."
    
    # 检查 firewalld 状态
    if ! sudo systemctl is-active --quiet firewalld; then
        log_warning "firewalld 未运行，正在启动..."
        sudo systemctl start firewalld
        sudo systemctl enable firewalld
    fi
    
    # 添加端口
    sudo firewall-cmd --permanent --add-port="$port/tcp"
    
    if [ -n "$ip_range" ]; then
        # 添加 rich rule 限制 IP
        sudo firewall-cmd --permanent --add-rich-rule="rule family=\"ipv4\" source address=\"$ip_range\" port protocol=\"tcp\" port=\"$port\" accept"
        log_success "允许 $ip_range 访问端口 $port"
    fi
    
    sudo firewall-cmd --reload
    log_success "firewalld 配置完成"
    
    sudo firewall-cmd --list-ports | grep "$port"
}

# 配置 iptables
setup_iptables() {
    local port=$1
    local ip_range=$2
    
    log_info "配置 iptables..."
    
    if [ -n "$ip_range" ]; then
        # 允许特定 IP
        sudo iptables -A INPUT -p tcp --dport "$port" -s "$ip_range" -j ACCEPT
        sudo iptables -A INPUT -p tcp --dport "$port" -j DROP
        log_success "允许 $ip_range 访问端口 $port"
    else
        # 允许所有
        sudo iptables -A INPUT -p tcp --dport "$port" -j ACCEPT
        log_success "允许所有访问端口 $port"
    fi
    
    # 保存规则
    if command -v iptables-save &> /dev/null; then
        sudo mkdir -p /etc/iptables
        sudo iptables-save > /etc/iptables/rules.v4
        log_success "规则已保存"
    fi
    
    sudo iptables -L | grep "$port"
}

# 显示当前配置
show_current() {
    log_info "当前网络配置:"
    echo ""
    
    # IP 地址
    echo "本机 IP 地址:"
    ip addr show | grep "inet " | grep -v "127.0.0.1" | awk '{print "  " $2}'
    echo ""
    
    # 防火墙状态
    echo "防火墙状态:"
    local fw=$(detect_firewall)
    echo "  类型: $fw"
    
    case $fw in
        ufw)
            sudo ufw status | head -10
            ;;
        firewalld)
            sudo firewall-cmd --state
            echo "开放端口:"
            sudo firewall-cmd --list-ports
            ;;
        iptables)
            sudo iptables -L INPUT -n | head -20
            ;;
        *)
            log_warning "未检测到防火墙"
            ;;
    esac
    echo ""
    
    # 监听端口
    echo "监听的端口:"
    sudo ss -tulpn | grep -E "(moxing|llama|python)" | head -10 || echo "  无"
}

# 删除端口规则
delete_port() {
    local port=$1
    local fw=$(detect_firewall)
    
    log_info "删除端口 $port 的规则..."
    
    case $fw in
        ufw)
            sudo ufw delete allow "$port/tcp" 2>/dev/null || true
            ;;
        firewalld)
            sudo firewall-cmd --permanent --remove-port="$port/tcp" 2>/dev/null || true
            sudo firewall-cmd --reload
            ;;
        iptables)
            sudo iptables -D INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || true
            sudo iptables -D INPUT -p tcp --dport "$port" -j DROP 2>/dev/null || true
            ;;
    esac
    
    log_success "端口 $port 规则已删除"
}

# 主流程
main() {
    echo "=========================================="
    echo "  MoXing 端口配置工具"
    echo "=========================================="
    echo ""
    
    local command="${1:-help}"
    local port="${2:-8080}"
    local ip_range="${3:-}"
    
    case $command in
        setup|add)
            local fw=$(detect_firewall)
            log_info "检测到防火墙: $fw"
            
            case $fw in
                ufw)
                    setup_ufw "$port" "$ip_range"
                    ;;
                firewalld)
                    setup_firewalld "$port" "$ip_range"
                    ;;
                iptables)
                    setup_iptables "$port" "$ip_range"
                    ;;
                *)
                    log_error "未检测到支持的防火墙"
                    exit 1
                    ;;
            esac
            ;;
        delete|remove)
            delete_port "$port"
            ;;
        show|status)
            show_current
            ;;
        help|--help|-h|*)
            echo "用法:"
            echo "  $0 setup <port> [ip_range]  # 配置端口"
            echo "  $0 delete <port>            # 删除端口"
            echo "  $0 show                     # 显示状态"
            echo ""
            echo "示例:"
            echo "  $0 setup 8080                    # 允许所有访问 8080"
            echo "  $0 setup 8080 192.168.1.0/24     # 仅允许局域网"
            echo "  $0 setup 8080 10.0.0.0/8         # 仅允许私有网络"
            echo "  $0 delete 8080                   # 删除规则"
            echo "  $0 show                          # 查看状态"
            echo ""
            echo "注意:"
            echo "  - 使用 IP 范围可以限制访问来源"
            echo "  - 云服务器还需要配置安全组"
            echo "  - 路由器需要配置端口转发"
            ;;
    esac
}

main "$@"
