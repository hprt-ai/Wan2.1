import multiprocessing
import traceback
multiprocessing.set_start_method('spawn', force=True)  # 确保在任何导入前设置

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
import os
import shutil
from datetime import datetime
import argparse
from PIL import Image
import io
import logging
import sys
import warnings
import torch
import torch.distributed as dist
import asyncio
from typing import Dict, Optional, List
import uuid
import aiofiles
from fastapi.responses import JSONResponse
from fastapi import status
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import tempfile
import pickle
import json
import threading
import glob
import queue
import dataclasses
from enum import Enum, auto

warnings.filterwarnings('ignore')

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video, cache_image

# 加入openapi
app = FastAPI(openapi_tags=[
    {"name": "生成视频", "description": "从图片生成视频"}, 
    {"name": "生成文本", "description": "从文本生成视频"},
    {"name": "任务状态", "description": "查询任务状态"}
], openapi_url="/openapi.json")

# 创建输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建临时文件目录
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# 任务状态目录
TASK_DIR = "tasks"
os.makedirs(TASK_DIR, exist_ok=True)

# 全局配置
# 从图片生成视频的配置
I2V_CHECKPOINT_DIR = "./Wan2.1-I2V-14B-720P" 
I2V_TASK = "i2v-14B"

# 从文本生成视频的配置
T2V_CHECKPOINT_DIR = "./Wan2.1-T2V-14B"
T2V_TASK = "t2v-14B"

# 通用配置
SIZE = "832*480"
SAMPLE_STEPS = 40
SAMPLE_SHIFT = 3.0
SAMPLE_SOLVER = "unipc"
SAMPLE_GUIDE_SCALE = 5.0

# 任务超时时间（秒）
TASK_TIMEOUT = 1800  # 30分钟

# 文件读取块大小
CHUNK_SIZE = 1024 * 1024  # 1MB

# 文件读取超时时间
READ_TIMEOUT = 30  # 30秒

# 用于限制同时处理的请求数量
REQUEST_SEMAPHORE = asyncio.Semaphore(5)

# 视频文件缓存大小
VIDEO_CACHE_SIZE = 20

# 进程池
process_pool = None

# 任务类型枚举
class TaskType(Enum):
    IMAGE_TO_VIDEO = auto()
    TEXT_TO_VIDEO = auto()

# 任务数据类
@dataclasses.dataclass
class TaskData:
    task_id: str
    task_type: TaskType
    prompt: str
    image_path: Optional[str] = None
    save_file: str = ""
    created_at: str = ""

# 全局任务队列和状态
task_queue = queue.Queue()
current_task = None
queue_lock = threading.Lock()  # 保护任务队列和当前任务变量
worker_thread = None
worker_running = False

# 任务队列管理器
def task_queue_worker():
    """后台工作线程，从队列中获取任务并提交到进程池执行"""
    global process_pool, current_task, worker_running
    
    logging.info("任务队列工作线程已启动")
    worker_running = True
    task_running = False  # 添加额外的标志来跟踪任务执行状态
    
    while worker_running:
        try:
            # 从队列获取任务
            with queue_lock:
                if task_running:
                    # 如果任务正在执行，只需等待
                    time.sleep(0.5)
                    continue
                
                if task_queue.empty():
                    current_task = None
                else:
                    if current_task is None:  # 只有当前没有运行任务时才获取新任务
                        current_task = task_queue.get(False)
                        logging.info(f"从队列中获取新任务: {current_task.task_id}, 队列剩余任务: {task_queue.qsize()}")
                        task_running = True  # 标记任务即将执行
            
            # 如果没有任务，等待一会再检查
            if current_task is None:
                time.sleep(0.5)
                continue
            
            try:
                # 确保每次任务开始前都有一个干净的进程池
                recreate_process_pool()
                
                # 根据任务类型执行不同的处理
                if current_task.task_type == TaskType.IMAGE_TO_VIDEO:
                    logging.info(f"开始处理图像到视频任务: {current_task.task_id}")
                    
                    # 更新任务状态
                    update_task_status(current_task.task_id, {"status": "initializing"})
                    
                    # 提交任务到进程池
                    future = process_pool.submit(
                        process_generate_video_from_image,
                        current_task.image_path,
                        current_task.prompt,
                        current_task.save_file,
                        current_task.task_id
                    )
                    
                    # 等待任务完成
                    result = future.result()
                    logging.info(f"图像到视频任务完成: {current_task.task_id}, 结果: {result}")
                    
                elif current_task.task_type == TaskType.TEXT_TO_VIDEO:
                    logging.info(f"开始处理文本到视频任务: {current_task.task_id}")
                    
                    # 更新任务状态
                    update_task_status(current_task.task_id, {"status": "initializing"})
                    
                    # 提交任务到进程池
                    future = process_pool.submit(
                        process_generate_video_from_text,
                        current_task.prompt,
                        current_task.save_file,
                        current_task.task_id
                    )
                    
                    # 等待任务完成
                    result = future.result()
                    logging.info(f"文本到视频任务完成: {current_task.task_id}, 结果: {result}")
                
            except Exception as e:
                logging.error(f"处理任务 {current_task.task_id} 失败: {e}", exc_info=True)
                update_task_status(current_task.task_id, {
                    "status": "failed",
                    "error": f"Failed to process task: {str(e)}"
                })
            finally:
                # 任务完成后，重新创建进程池以释放显存
                recreate_process_pool()
                
                # 任务完成或失败，重置当前任务
                with queue_lock:
                    current_task = None
                    task_running = False  # 重置任务执行状态
                
        except Exception as e:
            logging.error(f"任务队列工作线程发生错误: {e}", exc_info=True)
            with queue_lock:
                task_running = False  # 确保出错时也重置任务状态
            time.sleep(1)  # 出错后等待一会再继续
            
    logging.info("任务队列工作线程已退出")

def start_queue_worker():
    """启动任务队列处理线程"""
    global worker_thread, worker_running
    
    if worker_thread is None or not worker_thread.is_alive():
        worker_running = True
        worker_thread = threading.Thread(target=task_queue_worker, daemon=True)
        worker_thread.start()
        logging.info("已启动任务队列处理线程")
    
def stop_queue_worker():
    """停止任务队列处理线程"""
    global worker_running
    worker_running = False
    logging.info("已发送停止信号到任务队列处理线程")

def add_task_to_queue(task: TaskData) -> int:
    """添加任务到队列，返回当前队列长度"""
    with queue_lock:
        # 计算队列位置（如果当前有任务在执行，位置+1）
        position = task_queue.qsize() + (1 if current_task is not None else 0)
        task_queue.put(task)
        logging.info(f"已添加任务到队列: {task.task_id}, 当前队列位置: {position}")
        return position

def get_queue_status() -> Dict:
    """获取队列状态信息"""
    with queue_lock:
        queue_size = task_queue.qsize()
        has_current = current_task is not None
        current_id = current_task.task_id if has_current else None
        
    return {
        "tasks_in_queue": queue_size,
        "current_task": current_id,
        "total_pending": queue_size + (1 if has_current else 0)
    }

# 文件锁 - 用于防止文件竞争条件
def file_lock(path):
    lock_path = f"{path}.lock"
    while os.path.exists(lock_path):
        time.sleep(0.1)
    with open(lock_path, 'w') as f:
        f.write(str(time.time()))
    return lock_path

def release_file_lock(lock_path):
    if os.path.exists(lock_path):
        os.remove(lock_path)

def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)]
    )

def _validate_args(task, size):
    assert task in WAN_CONFIGS, f"Unsupport task: {task}"
    assert size in SUPPORTED_SIZES[task], f"Unsupport size {size} for task {task}"

def check_task_timeout(task_id: str):
    """检查任务是否超时，但保留任务结果"""
    task_file = os.path.join(TASK_DIR, f"{task_id}.json")
    if not os.path.exists(task_file):
        return False
    
    try:
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        created_at = datetime.fromisoformat(task["created_at"])
        # 只检查状态为pending、processing或initializing的任务
        if (task["status"] in ["pending", "processing", "initializing"] and 
            (datetime.now() - created_at).total_seconds() > TASK_TIMEOUT):
            # 获取文件锁
            lock_path = file_lock(task_file)
            try:
                # 再次读取以确保其他进程没有修改
                with open(task_file, 'r') as f:
                    task = json.load(f)
                
                # 如果任务已有结果，不要覆盖它
                if task["result"] is not None:
                    return False
                
                # 只有当任务没有结果时才更新状态为超时
                if task["status"] in ["pending", "processing", "initializing"]:
                    # 更新状态
                    task["status"] = "failed"
                    task["error"] = "Task timeout"
                    
                    # 写回文件
                    with open(task_file, 'w') as f:
                        json.dump(task, f)
                    
                    return True
            finally:
                release_file_lock(lock_path)
    except Exception as e:
        logging.error(f"检查任务超时错误: {e}")
    
    return False

# 定期清理过期任务
async def cleanup_tasks():
    """定期清理过期任务"""
    while True:
        try:
            current_time = datetime.now()
            # 获取所有任务文件
            task_files = glob.glob(os.path.join(TASK_DIR, "*.json"))
            
            for task_file in task_files:
                try:
                    with open(task_file, 'r') as f:
                        task = json.load(f)
                    
                    created_at = datetime.fromisoformat(task["created_at"])
                    # 清理超过2小时的已完成任务和超时任务
                    if ((task["status"] in ["completed", "failed"]) and 
                        (current_time - created_at).total_seconds() > 7200):
                        # 尝试删除文件
                        lock_path = file_lock(task_file)
                        try:
                            os.remove(task_file)
                            logging.info(f"清理过期任务: {os.path.basename(task_file)}")
                        except:
                            pass
                        finally:
                            release_file_lock(lock_path)
                except Exception as inner_e:
                    logging.error(f"清理任务文件失败 {task_file}: {inner_e}")
                
            # 每10分钟清理一次
            await asyncio.sleep(600)
        except Exception as e:
            logging.error(f"Error in cleanup_tasks: {e}")
            await asyncio.sleep(600)

# 保存任务状态到文件
def save_task_status(task_id, status_data):
    """保存任务状态到文件"""
    task_file = os.path.join(TASK_DIR, f"{task_id}.json")
    logging.info(f"正在保存任务状态到文件: {task_file}")
    
    # 确保任务目录存在
    if not os.path.exists(TASK_DIR):
        try:
            os.makedirs(TASK_DIR, exist_ok=True)
            logging.info(f"创建任务目录: {TASK_DIR}")
        except Exception as e:
            logging.error(f"创建任务目录失败: {e}")
            return False
    
    # 尝试获取文件锁
    lock_path = file_lock(task_file)
    try:
        # 写入任务状态文件
        with open(task_file, 'w') as f:
            json.dump(status_data, f)
        
        # 验证文件是否成功写入
        if os.path.exists(task_file):
            file_size = os.path.getsize(task_file)
            logging.info(f"任务状态已保存，文件大小: {file_size} 字节")
            return True
        else:
            logging.error(f"任务状态文件保存失败，文件不存在")
            return False
    except Exception as e:
        logging.error(f"保存任务状态失败: {e}", exc_info=True)
        return False
    finally:
        release_file_lock(lock_path)

# 更新任务状态
def update_task_status(task_id, updates):
    """更新任务状态"""
    task_file = os.path.join(TASK_DIR, f"{task_id}.json")
    if not os.path.exists(task_file):
        return False
    
    lock_path = file_lock(task_file)
    try:
        with open(task_file, 'r') as f:
            task = json.load(f)
        
        # 更新状态
        task.update(updates)
        
        # 写回文件
        with open(task_file, 'w') as f:
            json.dump(task, f)
        
        return True
    except Exception as e:
        logging.error(f"更新任务状态失败 {task_id}: {e}")
        return False
    finally:
        release_file_lock(lock_path)

# 获取任务状态
def get_task_data(task_id):
    """获取任务状态"""
    task_file = os.path.join(TASK_DIR, f"{task_id}.json")
    if not os.path.exists(task_file):
        return None
    
    try:
        with open(task_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"读取任务状态失败 {task_id}: {e}")
        return None

# 在单独进程中执行的视频生成函数
def process_generate_video_from_image(image_path, prompt, save_file, task_id):
    """在单独进程中执行视频生成，这个函数不能使用任何全局变量"""
    try:
        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
        
        logging.info(f"开始处理视频生成任务，任务ID：{task_id}")
        
        # 首先检查任务是否已完成或已被标记为超时
        task_file = os.path.join(TASK_DIR, f"{task_id}.json")
        if os.path.exists(task_file):
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    
                if task_data["status"] in ["completed", "failed"] and task_data["result"] is not None:
                    logging.info(f"任务已完成或失败，但有结果，跳过处理: {task_id}")
                    return True
            except Exception as e:
                logging.error(f"读取任务状态失败: {e}")
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        logging.info(f"加载图片完成")
        # 更新任务状态
        update_task_status(task_id, {"status": "processing"})
        logging.info(f"任务状态已更新为processing")
        
        # 加载配置 - 使用I2V专用配置
        cfg = WAN_CONFIGS[I2V_TASK]
        logging.info(f"加载配置完成")
        # 创建WanI2V实例
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=I2V_CHECKPOINT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True,
        )
        logging.info(f"创建WanI2V实例完成")
        # 添加开始时间
        start_time = time.time()
        logging.info(f"添加开始时间完成")
        # 生成视频
        video = wan_i2v.generate(
            prompt,
            image,
            max_area=MAX_AREA_CONFIGS[SIZE],
            frame_num=81,
            shift=SAMPLE_SHIFT,
            sample_solver=SAMPLE_SOLVER,
            sampling_steps=SAMPLE_STEPS,
            guide_scale=SAMPLE_GUIDE_SCALE,
            seed=-1,
            offload_model=True
        )

        # 记录生成时间
        generation_time = time.time() - start_time
        logging.info(f"记录生成时间完成")
        # 保存视频
        if video is not None:
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            logging.info(f"保存视频完成")
            logging.info(f"视频生成完成，耗时: {generation_time:.2f}秒")
            
            # 更新任务状态，即使任务已超时也更新结果信息
            # 首先检查当前任务状态
            current_status = get_task_data(task_id)
            update_data = {
                "result": save_file,
                "generation_time": f"{generation_time:.2f}秒"
            }
            
            # 如果任务没有被标记为失败（超时），则正常更新状态
            if current_status and current_status.get("status") != "failed":
                update_data["status"] = "completed"
                
            update_task_status(task_id, update_data)
            logging.info(f"任务状态已更新，保留结果: {save_file}")
            return True
        else:
            logging.error("视频生成失败，wan_i2v.generate返回None")
            # 更新任务状态
            update_task_status(task_id, {
                "status": "failed",
                "error": "Video generation failed"
            })
            logging.info(f"任务状态已更新为failed")
            return False
            
    except Exception as e:
        print("trackstack ", traceback.format_exc())
        logging.error(f"Error in process_generate_video_from_image: {e}")
        # 更新任务状态
        update_task_status(task_id, {
            "status": "failed",
            "error": str(e)
        })
        logging.info(f"任务状态已更新为failed (异常)")
        return False

def process_generate_video_from_text(prompt, save_file, task_id):
    """在单独进程中执行视频生成，这个函数不能使用任何全局变量"""
    try:
        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
        
        logging.info(f"开始处理文本到视频生成任务，任务ID：{task_id}")
        
        # 首先检查任务是否已完成或已被标记为超时
        task_file = os.path.join(TASK_DIR, f"{task_id}.json")
        if os.path.exists(task_file):
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    
                if task_data["status"] in ["completed", "failed"] and task_data["result"] is not None:
                    logging.info(f"任务已完成或失败，但有结果，跳过处理: {task_id}")
                    return True
            except Exception as e:
                logging.error(f"读取任务状态失败: {e}")
        
        # 更新任务状态
        update_task_status(task_id, {"status": "processing"})
        logging.info(f"任务状态已更新为processing")
        
        # 加载配置 - 使用T2V专用配置
        cfg = WAN_CONFIGS[T2V_TASK]
        
        # 创建WanT2V实例
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=T2V_CHECKPOINT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True,
        )

        # 添加开始时间
        start_time = time.time()
        
        # 生成视频
        video = wan_t2v.generate(
            prompt,
            size=SIZE_CONFIGS[SIZE],
            frame_num=81,
            shift=SAMPLE_SHIFT,
            sample_solver=SAMPLE_SOLVER,
            sampling_steps=SAMPLE_STEPS,
            guide_scale=SAMPLE_GUIDE_SCALE,
            seed=-1,
            offload_model=True
        )

        # 记录生成时间
        generation_time = time.time() - start_time
        
        # 保存视频
        if video is not None:
            cache_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            
            logging.info(f"视频生成完成，耗时: {generation_time:.2f}秒")
            
            # 更新任务状态，即使任务已超时也更新结果信息
            # 首先检查当前任务状态
            current_status = get_task_data(task_id)
            update_data = {
                "result": save_file,
                "generation_time": f"{generation_time:.2f}秒"
            }
            
            # 如果任务没有被标记为失败（超时），则正常更新状态
            if current_status and current_status.get("status") != "failed":
                update_data["status"] = "completed"
                
            update_task_status(task_id, update_data)
            logging.info(f"任务状态已更新，保留结果: {save_file}")
            return True
        else:
            logging.error("视频生成失败，wan_t2v.generate返回None")
            # 更新任务状态
            update_task_status(task_id, {
                "status": "failed",
                "error": "Video generation failed"
            })
            logging.info(f"任务状态已更新为failed")
            return False
            
    except Exception as e:
        logging.error(f"Error in process_generate_video_from_text: {e}")
        # 更新任务状态
        update_task_status(task_id, {
            "status": "failed",
            "error": str(e)
        })
        logging.info(f"任务状态已更新为failed (异常)")
        return False

@app.post("/generate_video")
async def generate_video(image: UploadFile = File(...), prompt: str = None):
    """从图片生成视频的API端点"""
    # 如果图片为空，返回错误
    if image is None:
        print("图片为空")
        return {"error": "图片为空"}
    if prompt is None:
        print("prompt为空")
        return {"error": "prompt为空"}
    
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        logging.info(f"收到图片到视频请求，生成任务ID: {task_id}")
        
        # 读取上传的图片并保存到临时文件
        image_content = await image.read()
        image_path = os.path.join(TEMP_DIR, f"image_{task_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(image_content)
        
        # 生成保存文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
        
        # 初始化任务状态
        task_data = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "queue_position": None,
            "prompt": prompt,
            "image_path": image_path,
            "save_file": save_file,
            "task_type": "image_to_video"
        }
        
        # 保存任务状态
        save_task_status(task_id, task_data)
        
        # 创建任务对象
        task = TaskData(
            task_id=task_id,
            task_type=TaskType.IMAGE_TO_VIDEO,
            prompt=prompt,
            image_path=image_path,
            save_file=save_file,
            created_at=task_data["created_at"]
        )
        
        # 确保任务队列工作线程正在运行
        start_queue_worker()
        
        # 添加任务到队列
        position = add_task_to_queue(task)
        
        # 更新任务状态，包括队列位置
        update_task_status(task_id, {
            "queue_position": position
        })
        
        return {
            "status": "accepted",
            "task_id": task_id,
            "message": "Video generation task added to queue",
            "queue_position": position
        }
            
    except Exception as e:
        logging.error(f"Error in generate_video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_text")
async def generate_from_text(prompt: str):
    """从文本生成视频的API端点"""
    try:
        # 输出日志，帮助调试
        logging.info(f"收到文本到视频请求，prompt: {prompt}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        logging.info(f"为文本到视频请求生成任务ID: {task_id}")
        
        # 生成保存文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
        
        # 初始化任务状态
        task_data = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "queue_position": None,
            "prompt": prompt,
            "save_file": save_file,
            "task_type": "text_to_video"
        }
        
        # 保存任务状态
        save_task_status(task_id, task_data)
        logging.info(f"已保存任务状态到文件: {os.path.join(TASK_DIR, f'{task_id}.json')}")
        
        # 创建任务对象
        task = TaskData(
            task_id=task_id,
            task_type=TaskType.TEXT_TO_VIDEO,
            prompt=prompt,
            save_file=save_file,
            created_at=task_data["created_at"]
        )
        
        # 确保任务队列工作线程正在运行
        start_queue_worker()
        
        # 添加任务到队列
        position = add_task_to_queue(task)
        
        # 更新任务状态，包括队列位置
        update_task_status(task_id, {
            "queue_position": position
        })
        
        return {
            "status": "accepted",
            "task_id": task_id,
            "message": "Video generation task added to queue",
            "queue_position": position
        }
            
    except Exception as e:
        logging.error(f"Error in generate_from_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态的API端点"""
    try:
        logging.info(f"收到任务状态查询请求，任务ID: {task_id}")
        # 从文件系统获取任务状态
        task_file = os.path.join(TASK_DIR, f"{task_id}.json")
        logging.info(f"检查任务文件是否存在: {task_file}")
        
        if not os.path.exists(task_file):
            logging.warning(f"任务文件不存在: {task_file}")
            # 检查临时目录中是否存在该任务文件（兼容旧版本）
            old_task_file = os.path.join(TEMP_DIR, f"task_{task_id}.json")
            if os.path.exists(old_task_file):
                logging.info(f"在旧位置找到任务文件: {old_task_file}")
                # 读取旧文件并移动到新位置
                try:
                    with open(old_task_file, 'r') as f:
                        task_data = json.load(f)
                    save_task_status(task_id, task_data)
                    logging.info(f"已将任务文件从旧位置移动到新位置")
                except Exception as e:
                    logging.error(f"读取旧任务文件失败: {e}")
                    raise HTTPException(status_code=404, detail="Task not found")
            else:
                # 检查任务目录中的所有文件
                all_task_files = glob.glob(os.path.join(TASK_DIR, "*.json"))
                logging.info(f"当前任务目录中有 {len(all_task_files)} 个任务文件")
                raise HTTPException(status_code=404, detail="Task not found")
                
        task = get_task_data(task_id)
        if task is None:
            logging.warning(f"获取任务数据失败，任务ID: {task_id}")
            raise HTTPException(status_code=404, detail="Task not found")
        
        logging.info(f"成功获取任务状态，任务ID: {task_id}, 状态: {task['status']}")
        
        # 检查任务是否应该超时，但允许保留结果
        check_task_timeout(task_id)
        # 重新获取可能已更新的状态
        task = get_task_data(task_id)
        
        # 如果任务状态为pending，更新实时队列位置
        if task["status"] == "pending":
            # 检查当前任务ID
            with queue_lock:
                if current_task and current_task.task_id == task_id:
                    # 任务已经在处理中但状态文件还未更新
                    position = 0
                else:
                    # 检查任务在队列中的位置
                    position = 1  # 默认位置，考虑当前正在执行的任务
                    if current_task:  # 如果有当前任务
                        found = False
                        # 将队列转为列表以便遍历
                        tasks_list = list(task_queue.queue)
                        for i, t in enumerate(tasks_list):
                            if t.task_id == task_id:
                                position = i + 1  # +1因为有当前任务
                                found = True
                                break
                        if not found:
                            position = None  # 任务不在队列中
                    else:
                        position = None
                        
            # 如果队列位置变化，更新状态
            if position is not None and (task.get("queue_position") != position):
                update_task_status(task_id, {"queue_position": position})
                task["queue_position"] = position
        
        # 构建响应
        response = {
            "task_id": task_id,
            "status": task["status"],
            "result": task["result"],
            "error": task["error"],
            "created_at": task["created_at"],
            "generation_time": task.get("generation_time", "尚未完成")
        }
        
        # 添加队列位置信息
        if "queue_position" in task and task["queue_position"] is not None:
            response["queue_position"] = task["queue_position"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"获取任务状态时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 缓存视频文件信息
@lru_cache(maxsize=VIDEO_CACHE_SIZE)
def get_video_info(filepath):
    """获取视频文件信息并缓存"""
    if not os.path.exists(filepath):
        return None
    stats = os.stat(filepath)
    return {
        "size": stats.st_size,
        "mtime": stats.st_mtime
    }

@app.get("/video/{filename}")
async def get_video(filename: str):
    """获取生成的视频文件的API端点"""
    try:
        video_path = os.path.join(OUTPUT_DIR, filename)
        
        # 快速检查文件是否存在和文件信息
        video_info = get_video_info(video_path)
        if not video_info:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # 小文件直接使用FileResponse
        if video_info["size"] < 10 * 1024 * 1024:  # 小于10MB的文件
            return FileResponse(
                video_path, 
                media_type="video/mp4",
                filename=filename
            )
        
        # 使用异步文件读取，并设置超时
        async def iterfile():
            try:
                async with asyncio.timeout(READ_TIMEOUT):
                    async with aiofiles.open(video_path, 'rb') as f:
                        while chunk := await f.read(CHUNK_SIZE):  # 使用更大的块
                            yield chunk
            except asyncio.TimeoutError:
                logging.error(f"Timeout reading file {filename}")
                yield b""  # 发送空数据结束流

        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(video_info["size"])  # 添加内容长度头
            }
        )
    except Exception as e:
        logging.error(f"Error in get_video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue")
async def get_task_queue_status():
    """获取任务队列状态"""
    try:
        # 获取队列状态
        status = get_queue_status()
        
        # 返回状态信息
        return {
            "queue_size": status["tasks_in_queue"],
            "current_task": status["current_task"],
            "total_pending": status["total_pending"]
        }
    except Exception as e:
        logging.error(f"获取队列状态时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 启动事件处理
@app.on_event("startup")
async def startup_event():
    global process_pool
    
    # 检查和创建必要的目录
    required_dirs = [OUTPUT_DIR, TEMP_DIR, TASK_DIR]
    for d in required_dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d, exist_ok=True)
                logging.info(f"创建目录: {d}")
            except Exception as e:
                logging.error(f"创建目录 {d} 失败: {e}")
    
    # 检查目录权限
    for d in required_dirs:
        try:
            test_file = os.path.join(d, ".permission_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logging.info(f"目录 {d} 权限正常")
        except Exception as e:
            logging.error(f"目录 {d} 权限检查失败: {e}")
    
    # 记录当前工作目录
    cwd = os.getcwd()
    logging.info(f"当前工作目录: {cwd}")
    logging.info(f"任务目录的绝对路径: {os.path.abspath(TASK_DIR)}")
    
    # 恢复未完成的任务
    try:
        # 获取所有任务文件
        task_files = glob.glob(os.path.join(TASK_DIR, "*.json"))
        pending_tasks = []
        
        for task_file in task_files:
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                
                # 检查是否为挂起或进行中的任务
                if task_data["status"] in ["pending", "processing", "initializing"]:
                    task_id = os.path.basename(task_file).replace(".json", "")
                    logging.info(f"发现未完成的任务: {task_id}, 状态: {task_data['status']}")
                    
                    # 标记为恢复中
                    update_task_status(task_id, {
                        "status": "pending",
                        "error": "Task was interrupted and will be restarted"
                    })
                    
                    # 准备恢复任务
                    if "prompt" in task_data:
                        prompt = task_data["prompt"]
                        image_path = task_data.get("image_path")
                        
                        # 生成新的保存文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
                        
                        # 创建任务对象
                        if image_path and os.path.exists(image_path):
                            # 图像到视频任务
                            task = TaskData(
                                task_id=task_id,
                                task_type=TaskType.IMAGE_TO_VIDEO,
                                prompt=prompt,
                                image_path=image_path,
                                save_file=save_file,
                                created_at=task_data["created_at"]
                            )
                        else:
                            # 文本到视频任务
                            task = TaskData(
                                task_id=task_id,
                                task_type=TaskType.TEXT_TO_VIDEO,
                                prompt=prompt,
                                save_file=save_file,
                                created_at=task_data["created_at"]
                            )
                        
                        pending_tasks.append(task)
            except Exception as e:
                logging.error(f"处理未完成任务文件失败 {task_file}: {e}")
        
        # 初始化进程池
        process_pool = ProcessPoolExecutor(max_workers=1)
        
        # 启动任务队列工作线程
        start_queue_worker()
        
        # 将未完成的任务添加到队列
        for task in pending_tasks:
            add_task_to_queue(task)
            logging.info(f"已将未完成任务添加到队列: {task.task_id}")
    
    except Exception as e:
        logging.error(f"恢复未完成任务失败: {e}", exc_info=True)
        # 确保进程池和任务队列仍然启动
        process_pool = ProcessPoolExecutor(max_workers=1)
        start_queue_worker()
    
    # 启动清理任务
    asyncio.create_task(cleanup_tasks())
    logging.info("服务启动完成，进程池和任务队列已初始化")

# 关闭进程池
@app.on_event("shutdown")
async def shutdown_event():
    global process_pool
    
    # 停止任务队列工作线程
    stop_queue_worker()
    
    # 关闭进程池
    if process_pool:
        process_pool.shutdown(wait=False)
        process_pool = None
    logging.info("服务关闭，进程池和任务队列已销毁")

# 进程池管理
def recreate_process_pool():
    """重新创建进程池，用于确保任务间释放资源"""
    global process_pool
    
    # 关闭现有进程池
    if process_pool:
        try:
            process_pool.shutdown(wait=False)
            logging.info("已关闭旧进程池")
        except Exception as e:
            logging.error(f"关闭进程池时出错: {e}")
    
    # 创建新进程池
    process_pool = ProcessPoolExecutor(max_workers=1)
    logging.info("已创建新进程池")
    return process_pool

if __name__ == "__main__":
    _init_logging()
    # 使用单进程模式运行，确保任务队列在单一进程中
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8088,
        workers=1,  # 使用单进程模式，确保任务队列正常工作
        log_level="info"
    )
