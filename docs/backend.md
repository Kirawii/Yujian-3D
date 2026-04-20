# 4.1.3 后端实现

本项目后端主要的技术栈包括：选用基于 Python 的 FastAPI 作为后端框架，使用 PyTorch 进行模型推理，选用 PyRender 进行 3D 视频渲染，使用 SSE (Server-Sent Events) 实现实时进度推送。

后端主要提供 REST 风格的接口和 SSE 流式接口响应给前端。

## 后端架构

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   前端应用   │──────▶│  FastAPI    │──────▶│  SOKE 模型  │
│  (Web/小程序)│◀──────│   后端服务   │◀──────│ (PyTorch)   │
└─────────────┘      └──────┬──────┘      └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  PyRender   │
                     │  3D渲染引擎  │
                     └─────────────┘
```

## 技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 高性能异步框架，天然支持 SSE |
| 模型推理 | PyTorch + SOKE | 文本到手语姿态生成 |
| 3D 渲染 | PyRender | SMPL-X 模型渲染为视频 |
| 实时通信 | SSE | Server-Sent Events 推送进度 |
| 部署 | Uvicorn | ASGI 服务器，支持高并发 |

## 后端职责

### 1. 负责模型推理任务的调度与执行

文本到手语视频生成是耗时任务（约 10-20 秒），后端采用异步处理机制：

- **任务队列**：使用内存中的任务字典管理 pending/processing/completed 状态
- **异步生成**：通过 `BackgroundTasks` 在后台执行模型推理
- **实时进度**：使用 SSE (Server-Sent Events) 向客户端实时推送生成进度（0% → 100%）

```python
@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    task_id = create_task(request.text)
    background_tasks.add_task(generate_task, task_id, request.text)
    return {"task_id": task_id, "status": "pending"}

@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    return StreamingResponse(progress_generator(task_id), media_type="text/event-stream")
```

### 2. 负责视频文件的生成与存储

- **视频渲染**：使用 PyRender 将 SMPL-X 3D 模型渲染为 MP4 视频
- **本地存储**：生成的视频暂存于 `api/outputs/` 目录
- **文件分发**：通过 `/video/{task_id}` 端点提供视频下载
- **定期清理**：输出目录设置自动清理策略，防止磁盘占满

### 3. 负责与前端的数据交互

后端提供以下核心接口：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/generate` | POST | 提交文本生成任务 |
| `/status/{task_id}` | GET | 查询任务状态 |
| `/progress/{task_id}` | GET | SSE 实时进度流 |
| `/video/{task_id}` | GET | 下载生成的视频 |
| `/tasks` | GET | 列出历史任务 |

### 4. 负责跨域与性能优化

- **CORS 配置**：允许前端跨域访问
- **响应头优化**：SSE 端点禁用缓冲 (`X-Accel-Buffering: no`)，确保实时推送
- **连接保持**：使用 `keep-alive` 长连接减少握手开销

## 核心代码结构

```
api/
├── main.py          # FastAPI 主应用入口
├── start_server.py  # 服务启动脚本
└── outputs/         # 视频输出目录
```

## 部署方式

```bash
# 生产环境启动
./start_api.sh

# 或手动启动
uvicorn api.main:app --host 0.0.0.0 --port 6008
```

服务监听 6008 端口，通过 Nginx 反向代理映射到公网 HTTPS 地址。
