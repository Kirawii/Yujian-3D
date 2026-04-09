# 语见 (Yujian-3D)

基于 SOKE 的中文手语翻译系统，支持文本到 3D 手语视频的生成。

## 简介

语见是一个中文手语翻译系统，能够将中文文本转换为 3D 手语动画视频，帮助听障人士更好地获取信息。

## 功能特点

- 🎯 **文本到手语**：输入中文文本，生成 3D 手语视频
- 🚀 **实时进度**：SSE 流式传输，实时查看生成进度
- 🌐 **API 服务**：提供 HTTP API，方便前端集成
- 📹 **高质量渲染**：基于 SMPL-X 模型，渲染逼真的 3D 人物

## 快速开始

### 启动服务

```bash
./start_api.sh
```

### API 使用

**提交任务**
```bash
curl -X POST http://localhost:6008/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "num_samples": 1}'
```

**查看进度（SSE）**
```javascript
const eventSource = new EventSource('http://localhost:6008/progress/{task_id}');
eventSource.onmessage = (e) => console.log(JSON.parse(e.data));
```

**下载视频**
```bash
curl -O http://localhost:6008/video/{task_id}
```

## API 文档

详见 [API.md](./API.md)

## 公网访问

```
https://uu895901-9072-0273df24.westc.seetacloud.com:8443
```

## 技术栈

- **后端**: FastAPI + PyTorch
- **模型**: SOKE (Signs as Tokens)
- **渲染**: PyRender + SMPL-X
- **部署**: AutoDL + Nginx

## 许可证

MIT License
