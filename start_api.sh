#!/bin/bash
# SOKE API 启动脚本

echo "Starting SOKE Sign Language API..."

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate soke

# 启动服务
cd /root/autodl-tmp/lijiahao/SOKE
nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 6008 > /tmp/api_server.log 2>&1 &

echo "API server started on port 6008"
echo "Local: http://localhost:6008"
echo "Public: https://uu895901-9072-0273df24.westc.seetacloud.com:8443"
echo ""
echo "Check logs: tail -f /tmp/api_server.log"
