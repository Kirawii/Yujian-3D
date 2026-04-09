"""
SOKE 手语翻译 FastAPI 后端
支持: 文本 → 3D 手语动作生成
"""
import os
import sys
import uuid
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import numpy as np
from omegaconf import OmegaConf

# Set environment variables
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables for model
model = None
datamodule = None
device = None

# Task storage
tasks = {}


class GenerateRequest(BaseModel):
    text: str
    num_samples: int = 1


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    text: str
    output_video: Optional[str] = None
    error_message: Optional[str] = None


def load_model():
    """加载 SOKE 模型"""
    global model, datamodule, device

    import sys
    from mGPT.config import parse_args, get_module_config, instantiate_from_config
    from mGPT.data.build_data import build_data
    from omegaconf import OmegaConf

    print("[INFO] Loading SOKE model...")

    # Force load soke.yaml config
    sys.argv = ['', '--cfg', './configs/soke.yaml']
    cfg = parse_args(phase="demo")

    # Override checkpoint path
    cfg.TEST.CHECKPOINTS = './experiments/mgpt/SOKE'

    # Fix missing config keys
    if not hasattr(cfg, 'TRAIN') or cfg.TRAIN is None:
        cfg.TRAIN = {}
    cfg.TRAIN.STAGE = cfg.TRAIN.get('STAGE', 'demo')

    # Force CSL-Daily dataset (avoid HumanML3D path issues)
    cfg.DATASET.TYPE = 'CSL-Daily'
    cfg.DATASET.DATA_ROOT = './data/CSL-Daily'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.DEVICE = [0]
    cfg.ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"

    # Build data module
    print("[INFO] Loading data module...")
    datamodule = build_data(cfg)
    datamodule.setup(stage="test")

    # Build model
    print("[INFO] Building model...")
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['params']['cfg'] = cfg
    model_config['params']['datamodule'] = datamodule
    model = instantiate_from_config(model_config)

    # Load checkpoint
    ckpt_path = './experiments/mgpt/SOKE/checkpoints/last.ckpt'
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = model.to(device)
    model.eval()

    print("[INFO] Model loaded successfully!")


def render_video(feats: np.ndarray, output_path: str, text: str = "", task_id: str = None):
    """使用 pyrender 渲染 SMPL-X 视频 - 基于 vis_mesh.py 官方实现"""
    import pyrender
    import trimesh
    from moviepy import ImageSequenceClip
    from PIL import Image, ImageDraw, ImageFont
    from mGPT.utils.human_models import smpl_x, get_coord

    # Load mean/std for denormalization
    mean = torch.load('./data/CSL-Daily/mean.pt', weights_only=False).cuda()
    std = torch.load('./data/CSL-Daily/std.pt', weights_only=False).cuda()
    mean = mean[(3+3*11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3+3*11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)

    # Convert feats to vertices using official feats2joints logic
    T = feats.shape[0]
    feats_tensor = torch.from_numpy(feats).cuda().unsqueeze(0)  # [1, T, 133]

    # Denormalize
    features = feats_tensor * std + mean

    # Add zero pose prefix and get vertices
    zero_pose = torch.zeros(1, T, 36).cuda()
    shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
                            0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).cuda()
    shape_param = shape_param.repeat(1, T, 1).view(T, -1)

    features = torch.cat([zero_pose, features], dim=-1).view(T, -1)  # [T, 169]
    with torch.no_grad():
        vertices, joints = get_coord(
            root_pose=features[..., 0:3],
            body_pose=features[..., 3:66],
            lhand_pose=features[..., 66:111],
            rhand_pose=features[..., 111:156],
            jaw_pose=features[..., 156:159],
            shape=shape_param,
            expr=features[..., 159:169]
        )
    vertices = vertices.cpu().numpy()

    # Camera parameters from vis_mesh.py
    h, w = 512, 512
    cam_trans = np.array([-2.6177440e-03, 0.1, -13], dtype=np.float32)

    # Render frames
    frames = []

    # Background color (light gray-blue)
    bg_color = np.array([200, 210, 230], dtype=np.uint8)  # 浅灰蓝色背景

    for t in range(T):
        # Create colored background
        img = np.ones((h, w, 3), dtype=np.uint8) * bg_color

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices[t], faces=smpl_x.face)

        # Apply rotation (180 degrees around X axis)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        # Material - darker color for better contrast
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.3,
            alphaMode='OPAQUE',
            baseColorFactor=(0.3, 0.4, 0.6, 1.0)  # 深蓝色人物
        )

        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

        # Scene with ambient light
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(py_mesh, 'mesh')

        # Camera (from vis_mesh.py)
        camera_center = [1.0 * w / 2, 1.0 * h / 2]
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = cam_trans
        camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

        focal_length = 5000
        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        # Light (from vis_mesh.py)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
        light_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        scene.add(light, pose=light_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        # Composite: mesh on colored background
        rgb = rgb[:, :, :3].astype(np.float32)
        valid_mask = (depth > 0)[:, :, None]
        img = rgb * valid_mask + img * (1 - valid_mask)
        img = np.array(img, dtype=np.uint8)

        frames.append(img)

        if t % 10 == 0:
            # Update progress
            if task_id and task_id in tasks:
                progress = int((t / T) * 90) + 10  # 10-100%
                tasks[task_id]["render_progress"] = progress
            print(f"[RENDER] Frame {t}/{T}")

    # Save video
    clip = ImageSequenceClip(frames, fps=18)
    clip.write_videofile(output_path, fps=18, codec='libx264', audio=False, logger=None)
    clip.close()

    print(f"[INFO] Video saved to {output_path}")


def generate_task(task_id: str, text: str, num_samples: int = 1):
    """后台生成任务"""
    global model, device
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["render_progress"] = 5
        tasks[task_id]["started_at"] = datetime.now().isoformat()

        print(f"[TASK {task_id}] Generating for text: {text}")

        # Reload model if not loaded (background task may not have access to global)
        if model is None:
            print(f"[TASK {task_id}] Model not loaded, reloading...")
            load_model()

        # Prepare batch
        batch = {
            "text": [text] * num_samples,
            "length": [196],
        }

        # Generate
        with torch.no_grad():
            outputs = model(batch, task="t2m")

        print(f"[DEBUG] Model output type: {type(outputs)}")
        print(f"[DEBUG] Model output keys: {outputs.keys() if isinstance(outputs, dict) else 'N/A'}")
        print(f"[DEBUG] Model: {model}")

        if outputs is None:
            raise ValueError("Model returned None")

        feats = outputs.get("feats") if isinstance(outputs, dict) else None
        if feats is None:
            raise ValueError(f"Model output does not contain 'feats'. Keys: {outputs.keys() if isinstance(outputs, dict) else 'N/A'}")
        feats_np = feats.cpu().numpy()

        # Render each sample
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        video_paths = []
        for i in range(num_samples):
            output_path = str(output_dir / f"{task_id}_sample{i}.mp4")
            render_video(feats_np[i], output_path, text, task_id)
            video_paths.append(f"{task_id}_sample{i}.mp4")

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["completed_at"] = datetime.now().isoformat()
        tasks[task_id]["output_videos"] = video_paths

        print(f"[TASK {task_id}] Completed!")

    except Exception as e:
        error_msg = traceback.format_exc()
        # Write to file for debugging
        with open(f'/tmp/task_{task_id}_error.log', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{error_msg}")
        print(f"[TASK {task_id}] Error: {error_msg}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error_message"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("[INFO] Starting SOKE Sign Language API...")
    try:
        load_model()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[WARNING] API will start without model loaded")
    yield
    print("[INFO] Shutting down...")


app = FastAPI(
    title="SOKE Sign Language API",
    description="Text to 3D Sign Language Generation",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/videos", StaticFiles(directory=str(Path(__file__).parent / "outputs")), name="videos")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    提交文本生成手语视频任务

    - **text**: 要翻译的中文文本
    - **num_samples**: 生成样本数量 (默认 1)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_id = str(uuid.uuid4())[:8]

    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "text": request.text,
        "num_samples": request.num_samples,
        "output_videos": [],
        "error_message": None
    }

    background_tasks.add_task(generate_task, task_id, request.text, request.num_samples)

    return GenerateResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted successfully. Use /status/{task_id} to check progress."
    )


@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_status(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        text=task["text"],
        output_video=task["output_videos"][0] if task["output_videos"] else None,
        error_message=task.get("error_message")
    )


@app.get("/video/{task_id}")
async def get_video(task_id: str, sample: int = Query(0, ge=0)):
    """
    获取生成的视频文件

    - **task_id**: 任务ID
    - **sample**: 样本索引 (默认 0)
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not completed. Status: {task['status']}")

    if sample >= len(task["output_videos"]):
        raise HTTPException(status_code=404, detail="Sample index out of range")

    video_path = Path(__file__).parent / "outputs" / task["output_videos"][sample]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"sign_language_{task_id}_sample{sample}.mp4"
    )


@app.get("/tasks")
async def list_tasks(limit: int = Query(10, le=100)):
    """列出最近的任务"""
    sorted_tasks = sorted(tasks.values(), key=lambda x: x["created_at"], reverse=True)
    return {"tasks": sorted_tasks[:limit]}


@app.get("/")
async def root():
    return {
        "name": "SOKE Sign Language API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "POST /generate": "Submit text to generate sign language video",
            "GET /status/{task_id}": "Check task status",
            "GET /video/{task_id}": "Download generated video",
            "GET /tasks": "List recent tasks",
            "GET /progress/{task_id}": "SSE real-time progress stream"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)


# SSE progress streaming
from fastapi.responses import StreamingResponse
import asyncio

async def progress_stream(task_id: str):
    """SSE stream for task progress"""
    last_status = None
    last_progress = -1

    while True:
        if task_id not in tasks:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            break

        task = tasks[task_id]
        current_status = task["status"]

        # Calculate progress percentage
        progress = 0
        if current_status == "pending":
            progress = 0
        elif current_status == "processing":
            # Estimate progress based on render progress if available
            progress = task.get("render_progress", 10)
        elif current_status == "completed":
            progress = 100
        elif current_status == "failed":
            progress = -1

        # Only send update if status or progress changed
        if current_status != last_status or progress != last_progress:
            data = {
                "task_id": task_id,
                "status": current_status,
                "progress": progress,
                "text": task.get("text", ""),
                "message": {
                    "pending": "等待处理...",
                    "processing": f"生成中... {progress}%",
                    "completed": "生成完成！",
                    "failed": f"生成失败: {task.get('error_message', '未知错误')}"
                }.get(current_status, current_status)
            }
            # SSE format with double newline to ensure flush
            sse_data = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield sse_data

            last_status = current_status
            last_progress = progress

            # End stream if completed or failed
            if current_status in ["completed", "failed"]:
                break

        await asyncio.sleep(0.2)  # Check every 200ms for more responsive updates


@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    SSE endpoint for real-time task progress
    
    Frontend usage:
        const eventSource = new EventSource(`/progress/${task_id}`);
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.progress + '%', data.message);
        };
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return StreamingResponse(
        progress_stream(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            "Content-Type": "text/event-stream; charset=utf-8",
        }
    )
