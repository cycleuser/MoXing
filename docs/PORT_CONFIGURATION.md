# 端口配置指南

## 概述

MoXing Ollama Runner 支持多种网络配置模式，从本地访问到完全对外开放。

## 主机绑定选项

### 1. 仅本地访问 (默认)

```bash
moxing ollama serve gemma4:31b --host 127.0.0.1
```

- **适用场景**: 仅本机访问，最安全
- **访问地址**: http://127.0.0.1:8080
- **外部访问**: ❌ 不允许

### 2. 局域网访问

```bash
moxing ollama serve gemma4:31b --host 0.0.0.0
```

- **适用场景**: 同一网络内其他设备访问
- **访问地址**: http://<本机IP>:8080
- **外部访问**: ⚠️ 局域网内可访问

### 3. 指定网卡访问

```bash
# 绑定到特定 IP
moxing ollama serve gemma4:31b --host 192.168.1.100
```

- **适用场景**: 多网卡服务器，指定特定网络
- **访问地址**: http://192.168.1.100:8080

## 端口配置

### 默认端口

```bash
# 使用默认端口 8080
moxing ollama serve gemma4:31b
```

### 自定义端口

```bash
# 指定端口
moxing ollama serve gemma4:31b --port 9000

# 高端口（1024以上）
moxing ollama serve gemma4:31b --port 18080

# 低端口（需要 root）
sudo moxing ollama serve gemma4:31b --port 80
```

### 自动寻找可用端口

```bash
moxing ollama serve gemma4:31b --port 0
```

## 完整示例

### 本地开发

```bash
# 本地，默认端口
moxing ollama serve gemma4:31b

# 本地，指定端口
moxing ollama serve gemma4:31b --port 8888
```

### 局域网共享

```bash
# 局域网可访问，默认端口
moxing ollama serve gemma4:31b --host 0.0.0.0

# 局域网可访问，指定端口
moxing ollama serve gemma4:31b --host 0.0.0.0 --port 8080

# 指定设备 + 局域网
moxing ollama serve gemma4:31b -b cuda -d gpu0 --host 0.0.0.0 --port 8080
```

### 多实例部署

```bash
# 实例 1: CUDA GPU 0, 端口 8080
moxing ollama serve gemma4:31b -b cuda -d gpu0 --port 8080 &

# 实例 2: CUDA GPU 1, 端口 8081
moxing ollama serve gemma4:31b -b cuda -d gpu1 --port 8081 &

# 实例 3: ROCm GPU 0, 端口 8082
moxing ollama serve llama3:70b -b rocm -d gpu0 --port 8082 &
```

## 防火墙配置

### Linux (ufw)

```bash
# 允许特定端口
sudo ufw allow 8080/tcp

# 允许端口范围
sudo ufw allow 8000:9000/tcp

# 仅允许特定 IP
sudo ufw allow from 192.168.1.0/24 to any port 8080
```

### Linux (firewalld)

```bash
# 允许端口
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# 允许服务（如果在同一机器）
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload
```

### Docker (如果使用)

```bash
# 映射端口到主机
docker run -p 8080:8080 moxing:latest moxing ollama serve gemma4:31b --host 0.0.0.0

# 映射到主机特定端口
docker run -p 18080:8080 moxing:latest moxing ollama serve gemma4:31b --host 0.0.0.0
```

## 安全建议

### 1. 不要直接对外开放

❌ **不推荐**:
```bash
# 直接绑定到公网接口
moxing ollama serve gemma4:31b --host 0.0.0.0 --port 80
```

### 2. 使用反向代理

✅ **推荐** - 使用 Nginx:

```nginx
# /etc/nginx/sites-available/moxing
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
        
        # 超时设置（长连接）
        proxy_read_timeout 86400;
    }
}
```

✅ **推荐** - 使用 Caddy:

```caddy
# Caddyfile
your-domain.com {
    reverse_proxy localhost:8080
}
```

### 3. 添加认证

#### Nginx + 基础认证

```bash
# 创建密码文件
sudo htpasswd -c /etc/nginx/.htpasswd username

# nginx 配置
location / {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://127.0.0.1:8080;
}
```

#### API Key 认证（应用层）

使用 Python 包装器：

```python
# api_proxy.py
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)
API_KEY = os.environ.get('API_KEY', 'your-secret-key')
MOXING_URL = 'http://127.0.0.1:8080'

@app.before_request
def check_auth():
    if request.headers.get('Authorization') != f'Bearer {API_KEY}':
        return jsonify({'error': 'Unauthorized'}), 401

@app.route('/<path:path>')
def proxy(path):
    resp = requests.request(
        method=request.method,
        url=f'{MOXING_URL}/{path}',
        headers={k:v for k,v in request.headers if k.lower() != 'host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    return resp.content, resp.status_code, resp.headers.items()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
```

### 4. 使用 VPN/内网穿透

#### Tailscale (推荐)

```bash
# 安装 Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 启动
sudo tailscale up

# 获取 IP
tailscale ip -4

# 其他 Tailscale 网络设备可访问
moxing ollama serve gemma4:31b --host 0.0.0.0
```

#### frp (内网穿透)

```ini
# frpc.ini (客户端)
[common]
server_addr = your-server.com
server_port = 7000
token = your-token

[moxing]
type = tcp
local_ip = 127.0.0.1
local_port = 8080
remote_port = 8080
```

### 5. 限制访问 IP

```bash
# 使用 iptables
sudo iptables -A INPUT -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP

# 保存规则
sudo iptables-save > /etc/iptables/rules.v4
```

## 监控访问日志

```bash
# 查看实时访问
tail -f ~/.moxing/logs/server.log

# 查看特定端口连接
sudo netstat -tulpn | grep 8080

# 查看当前连接
ss -tulpn | grep 8080
```

## 常见问题

### Q: 绑定 0.0.0.0 后外网无法访问？

检查：
1. 防火墙是否放行端口
2. 路由器是否配置了端口转发
3. 云服务器安全组是否放行

### Q: 端口被占用？

```bash
# 查找占用进程
sudo lsof -i :8080

# 或
sudo netstat -tulpn | grep 8080

# 使用自动端口选择
moxing ollama serve gemma4:31b --port 0
```

### Q: 如何查看当前运行的服务？

```bash
# 查看 moxing 进程
ps aux | grep moxing

# 查看端口监听
sudo ss -tulpn | grep moxing
```

## 推荐配置总结

| 场景 | 命令 | 安全级别 |
|------|------|----------|
| 本地开发 | `moxing ollama serve gemma4:31b` | ⭐⭐⭐⭐⭐ |
| 局域网共享 | `moxing ollama serve gemma4:31b --host 0.0.0.0` | ⭐⭐⭐ |
| 团队共享 | `moxing ollama serve gemma4:31b --host 0.0.0.0` + 防火墙限制 | ⭐⭐⭐⭐ |
| 远程访问 | + VPN/Tailscale | ⭐⭐⭐⭐⭐ |
| 公网服务 | + Nginx + 认证 | ⭐⭐⭐⭐⭐ |
