# 部署 Runbook（M1 测试服务器）

> 目标：把 PriceRadar 部署到国内云服务器 —— 快照采集 24/7 自动跑（解决"关机断采"）、Web 可外网访问、为 M3 小程序准备已备案 API 域名。
> 预计首次投入：半天 + ¥150~350/年（学生机价格）。

---

## 0. 决策点（部署前先定）

| 项 | 建议 | 说明 |
|---|---|---|
| 云厂商 | 阿里云 / 腾讯云（**学生优惠**） | 需学生认证；轻量服务器 2C2G 足够（本项目单机跑全部：Web + 快照 cron） |
| 域名 | 任意 `.com/.cn`，**与服务器同一厂商购买** | 同厂商一站式备案最省事 |
| 备案主体 | **先个人备案即可** | 小程序服务器域名只要求"域名已备案"，不要求主体与小程序一致；等个体工商户下来后新域名再用其备案 |
| HTTPS | Let's Encrypt（certbot 免费） | 小程序要求 API 必须 HTTPS |

⚠️ 备案期间（1~3 周）国内服务器 80/443 端口不可对外服务——先用 `IP:非标端口` 或本机开发，不阻塞。

---

## 1. 购买清单

- [ ] 轻量应用服务器（Ubuntu 22.04/24.04，2C2G，学生机 ~¥100-300/年），记录公网 IP
- [ ] 域名（~¥30-60/年），完成 DNS 解析 A 记录 → 服务器 IP
- [ ] （可选）安全组/防火墙：仅放行 22 / 80 / 443

## 2. ICP 备案（与后续开发并行）

- [ ] 厂商控制台 → 备案系统 → 按引导提交（个人：身份证 + 电子化核验刷脸 + 网站信息）
- [ ] 网站名称/服务内容：如"个人技术工具"，如实填写；备案通过前不要对外提供 80/443 服务
- [ ] 通过后：在 nginx/防火墙放行，域名即可访问

## 3. 服务器环境（首次）

```bash
# 3.1 基础环境
sudo apt update && sudo apt install -y python3.12-venv python3-pip nginx git
sudo useradd -m -s /bin/bash priceradar          # 专用运行用户

# 3.2 拉代码（私有仓库用 deploy key；或用 rsync/scp 上传）
sudo mkdir -p /opt/priceradar && sudo chown priceradar:priceradar /opt/priceradar
sudo -u priceradar git clone <你的仓库地址> /opt/priceradar
cd /opt/priceradar

# 3.3 虚拟环境 + 依赖（清华镜像加速）
sudo -u priceradar python3 -m venv .venv
sudo -u priceradar .venv/bin/pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

# 3.4 凭证：本地 .env 用 scp 上传（不进 git！）
scp ./.env priceradar@<服务器IP>:/opt/priceradar/.env
sudo chmod 600 /opt/priceradar/.env && sudo chown priceradar:priceradar /opt/priceradar/.env
```

## 4. systemd 服务

`/etc/systemd/system/priceradar.service`：

```ini
[Unit]
Description=PriceRadar Web
After=network.target

[Service]
User=priceradar
WorkingDirectory=/opt/priceradar
ExecStart=/opt/priceradar/.venv/bin/python -m uvicorn priceradar.service.api:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now priceradar
curl -s http://127.0.0.1:8000/api/watchlist-status   # 本机健康检查
```

## 5. nginx + HTTPS

`/etc/nginx/sites-available/priceradar`（`server_name` 换成你的域名）：

```nginx
server {
    listen 80;
    server_name prd.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;   # 含 LLM/检索的长请求
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/priceradar /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d prd.example.com     # 自动配置 HTTPS + 续期
```

## 6. 快照定时任务（部署的核心收益）

```bash
sudo -u priceradar crontab -e
# 每天 09:07 采集（错开整点；电脑关机问题从此消失）
7 9 * * * cd /opt/priceradar && .venv/bin/python -m priceradar.jobs.snapshot >> /var/log/priceradar-snapshot.log 2>&1
```

## 7. 日常更新

```bash
sudo -u priceradar bash /opt/priceradar/scripts/deploy.sh
```

（脚本内容：git pull → 装依赖 → 跑测试护栏 → 重启服务 → 健康检查；测试不过则中止）

## 8. 小程序前置（M3）

- [ ] 微信公众平台 → 小程序 → 开发设置 → **服务器域名（request 合法域名）**填 `https://prd.example.com`
- [ ] 隐私政策页（PIPL）：可用 Web 端加 `/privacy` 静态页后同域名引用
- [ ] 个体工商户注册（并行办理）

## 9. 成本

| 项 | 年成本 |
|---|---|
| 轻量服务器（学生） | ¥100-300 |
| 域名 | ¥30-60 |
| SSL | ¥0（Let's Encrypt） |
| 合计 | **约 ¥130-360/年** |

## 10. 安全清单

- [ ] 防火墙仅放 22/80/443；uvicorn 只监听 127.0.0.1（nginx 反代）
- [ ] `.env` 权限 600、不进 git；服务器用户无 sudo
- [ ] nginx 加基础限流（`limit_req_zone`），防 API 配额被恶意刷
- [ ] 定期 `journalctl -u priceradar` 看错误；快照日志轮转（logrotate）
