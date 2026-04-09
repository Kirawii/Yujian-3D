# SOKE Sign Language API

基于 SOKE (Sign Language Translation with Kinetic Energy) 的手语翻译 FastAPI 后端服务。

## 功能

- **文本 → 3D 手语动作**: 将中文文本转换为 SMPL-X 格式的 3D 手语动作
- **视频渲染**: 使用 pyrender 渲染 3D 人物视频
- **异步任务**: 支持后台任务处理

## API 端点

### 1. 提交生成任务
```bash
POST /generate

Request:
{
    "text": "你好",
    "num_samples": 1
}

Response:
{
    "task_id": "abc123",
    "status": "pending",
    "message": "Task submitted successfully. Use /status/{task_id} to check progress."
}
```

### 2. 查询任务状态
```bash
GET /status/{task_id}

Response:
{
    "task_id": "abc123",
    "status": "completed",
    "created_at": "2024-01-01T00:00:00",
    "completed_at": "2024-01-01T00:00:30",
    "text": "你好",
    "output_video": "abc123_sample0.mp4",
    "error_message": null
}
```

### 3. 获取视频文件
```bash
GET /video/{task_id}?sample=0

返回: MP4 视频文件
```

### 4. 列出任务
```bash
GET /tasks?limit=10
```

## 启动服务

### 启动服务 (端口 6006)
```bash
cd /root/autodl-tmp/lijiahao/SOKE
python api/start_server.py
```

服务将绑定到 `0.0.0.0:6006`，通过以下地址访问：
- **公网地址**: https://u895901-9072-0273df24.westc.seetacloud.com:8443

## 使用示例

### Python 客户端
```python
import requests
import time

# 提交任务 (本地或公网)
# BASE_URL = "https://u895901-9072-0273df24.westc.seetacloud.com:8443"  # 公网
BASE_URL = "http://127.0.0.1:6006"  # 本地

response = requests.post(f"{BASE_URL}/generate", json={
    "text": "我很高兴见到你",
    "num_samples": 1
})
data = response.json()
task_id = data["task_id"]
print(f"Task ID: {task_id}")

# 轮询状态
while True:
    status = requests.get(f"http://localhost:8000/status/{task_id}").json()
    print(f"Status: {status['status']}")
    
    if status["status"] == "completed":
        # 下载视频
        video = requests.get(f"http://localhost:8000/video/{task_id}")
        with open("output.mp4", "wb") as f:
            f.write(video.content)
        print("Video saved!")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error_message']}")
        break
    
    time.sleep(2)
```

### cURL
```bash
# 提交任务
curl -X POST "http://127.0.0.1:6006/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "你好", "num_samples": 1}'

# 查询状态
curl "http://127.0.0.1:6006/status/abc123"

# 下载视频
curl "http://127.0.0.1:6006/video/abc123" -o output.mp4
```

## 模型配置

- **配置文件**: `configs/soke.yaml`
- **检查点**: `experiments/mgpt/SOKE/checkpoints/last.ckpt`
- **数据**: CSL-Daily, How2Sign, Phoenix-2014T

## 技术细节

- **框架**: FastAPI + PyTorch Lightning
- **3D 渲染**: pyrender + SMPL-X
- **视频编码**: moviepy
- **人体模型**: SMPL-X (包含身体 + 手部)

## 注意事项

1. 首次启动会加载模型，可能需要几分钟
2. 视频生成时间取决于动作长度，通常需要 1-3 分钟
3. 确保 CUDA 可用以加速生成
4. 生成的视频保存在 `api/outputs/` 目录

## 文件结构

```
api/
├── main.py           # FastAPI 主应用
├── start_server.py   # 启动脚本
├── outputs/          # 生成的视频
├── models/           # (可选) 存放模型文件
└── README.md         # 本文档
```
