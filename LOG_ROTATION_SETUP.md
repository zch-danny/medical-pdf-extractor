# 日志轮转配置说明

## 应用日志 (已配置)
应用日志已经通过 loguru 配置了自动轮转：
- 位置: `logs/app_*.log` 和 `logs/error_*.log`
- 轮转策略: 文件达到 100MB 或每天午夜
- 保留期: 应用日志 7 天，错误日志 30 天
- 压缩: 自动 gzip 压缩

## vLLM 服务器日志轮转设置

### 方法 1: 使用 logrotate (推荐)
```bash
# 测试配置
logrotate -d logrotate_vllm.conf

# 手动执行一次轮转
logrotate -f logrotate_vllm.conf

# 添加到 crontab（每天检查）
echo "0 2 * * * /usr/sbin/logrotate /root/pdf_summarization_deploy_20251225_093847/logrotate_vllm.conf" | crontab -
```

### 方法 2: 修改启动脚本使用管道
修改 vLLM 启动脚本，将日志通过管道传递给 logger：
```bash
python -m vllm.entrypoints.openai.api_server ... 2>&1 | \
    logger -t vllm -p local0.info --size 100M --rotate 5
```

## 清理当前大日志文件
```bash
# 备份当前日志
mv vllm_server.log vllm_server.log.$(date +%Y%m%d)
gzip vllm_server.log.$(date +%Y%m%d)

# 或者直接清空（如果不需要历史）
> vllm_server.log
```
