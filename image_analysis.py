import os
import json
import hashlib
from pathlib import Path
import time
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
import base64
from openai import OpenAI
import uuid
from requests_toolbelt.multipart.encoder import MultipartEncoder

# 配置
INPUT_DIR = "dataset"  # 输入图片目录
OUTPUT_DIR = "output"  # 输出结果目录
CACHE_FILE = "output/cache.json"  # 缓存文件路径
MAX_WORKERS = 4  # 并行处理的最大线程数
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 支持的图片格式
VIDEO_API_BASE = "http://36.251.249.6:8088"  # 视频生成API基础URL
VIDEO_CHECK_INTERVAL = 30  # 视频生成状态检查间隔（秒）
# 不再需要最大检查次数，现在检查直到所有任务完成或失败

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def ensure_dirs():
    """确保输出和缓存目录存在"""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

def ensure_image_dir(image_name):
    """为每张图片创建单独的子文件夹"""
    # 移除文件扩展名
    base_name = os.path.splitext(image_name)[0]
    # 创建子文件夹
    image_dir = os.path.join(OUTPUT_DIR, base_name)
    Path(image_dir).mkdir(exist_ok=True)
    return image_dir

def load_cache() -> Dict[str, Any]:
    """加载缓存文件"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"缓存文件损坏，创建新缓存")
            return {}
    return {}

def save_cache(cache: Dict[str, Any]):
    """保存缓存文件"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_image_hash(image_path: str) -> str:
    """计算图片的hash值作为唯一标识"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_image_base64(image_path: str) -> str:
    """将图片转换为base64编码"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_image(image_path: str, cache: Dict[str, Any]) -> Optional[Dict]:
    """分析图片内容，生成描述"""
    try:
        # 计算图片hash
        image_hash = get_image_hash(image_path)
        file_name = os.path.basename(image_path)
        
        # 检查缓存
        if image_hash in cache:
            cache_entry = cache[image_hash]
            # 如果是简单字符串格式的旧缓存，转换为新格式
            if isinstance(cache_entry, str):
                cache[image_hash] = {
                    "file_name": file_name,
                    "image_path": image_path,
                    "image_recognition": None,
                    "optimized_prompt": None,
                    "output_dir": ensure_image_dir(file_name),
                    "video_task_id": None,
                    "video_path": None,
                    "video_status": None
                }
            # 如果已经有图像识别结果，则跳过
            elif isinstance(cache_entry, dict) and cache_entry.get("image_recognition"):
                print(f"图像识别结果已存在: {file_name}")
                return None
            
        else:
            # 创建新的缓存条目
            output_dir = ensure_image_dir(file_name)
            cache[image_hash] = {
                "file_name": file_name,
                "image_path": image_path,
                "image_recognition": None,
                "optimized_prompt": None,
                "output_dir": output_dir,
                "video_task_id": None,
                "video_path": None,
                "video_status": None
            }
        
        print(f"正在分析图片: {file_name}")
        
        # 调用API分析图片
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容，包括图片中的主要对象、场景、动作和可能的情感"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{get_image_base64(image_path)}"}}
                ]
            }]
        )
        
        # 解析结果
        result = completion.model_dump()
        image_description = result["choices"][0]["message"]["content"]
        
        # 更新缓存
        cache[image_hash]["image_recognition"] = image_description
        
        # 添加延迟避免API限制
        time.sleep(0.5)
        
        return result
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None

def optimize_prompt(image_hash: str, cache: Dict[str, Any]) -> Optional[str]:
    """使用AI优化生成的prompt"""
    try:
        cache_entry = cache.get(image_hash)
        if not cache_entry:
            print(f"缓存中未找到图片 {image_hash}")
            return None
            
        # 如果已经优化了prompt，则跳过
        if cache_entry.get("optimized_prompt"):
            print(f"优化Prompt已存在: {cache_entry['file_name']}")
            return cache_entry["optimized_prompt"]
            
        # 检查是否有图像识别结果
        image_description = cache_entry.get("image_recognition")
        if not image_description:
            print(f"未找到图像识别结果: {cache_entry['file_name']}")
            return None
            
        print(f"正在生成优化Prompt: {cache_entry['file_name']}")
        
        # 调用API优化prompt
        completion = client.chat.completions.create(
            model="qwen-max",  # 使用文本模型
            messages=[{
                "role": "user",
                "content": f"根据以下图像描述，生成一个用于文生图AI的prompt，使其精确、有创意，但保持简洁（不超过80单词）。使用英文，添加适当的艺术风格、光照、构图、动作等细节:\n\n{image_description}"
            }]
        )
        
        # 获取优化后的prompt
        optimized_prompt = completion.choices[0].message.content
        
        # 更新缓存
        cache_entry["optimized_prompt"] = optimized_prompt
        
        # 添加延迟避免API限制
        time.sleep(0.5)
        
        return optimized_prompt
    except Exception as e:
        print(f"优化Prompt时出错: {str(e)}")
        return None

def generate_video(image_hash: str, cache: Dict[str, Any]) -> Optional[str]:
    """调用API生成视频"""
    try:
        cache_entry = cache.get(image_hash)
        if not cache_entry:
            print(f"缓存中未找到图片 {image_hash}")
            return None
            
        # 检查视频状态
        video_status = cache_entry.get("video_status")
        if video_status == "completed" and cache_entry.get("video_path"):
            print(f"视频已生成: {cache_entry['file_name']}")
            return cache_entry["video_path"]
            
        if video_status == "processing" and cache_entry.get("video_task_id"):
            print(f"视频正在生成中: {cache_entry['file_name']}, 任务ID: {cache_entry['video_task_id']}")
            return None
            
        # 检查是否有优化后的prompt
        optimized_prompt = cache_entry.get("optimized_prompt")
        if not optimized_prompt:
            print(f"未找到优化后的Prompt: {cache_entry['file_name']}")
            return None
            
        image_path = cache_entry.get("image_path")
        if not image_path or not os.path.exists(image_path):
            print(f"图片路径无效: {image_path}")
            return None
        
        print(f"正在提交视频生成任务: {cache_entry['file_name']}")
        
        # 准备API请求
        url = f"{VIDEO_API_BASE}/generate_video"
        
        # 读取图片文件
        with open(image_path, 'rb') as img_file:
            # 最终尝试方案：在多个位置传递prompt
            image_content = img_file.read()
            
            # 准备文件参数
            files = {'image': (os.path.basename(image_path), image_content, 'image/jpeg')}
            
            # 同时在表单数据、URL参数和JSON中提供prompt
            data = {'prompt': optimized_prompt}
            params = {'prompt': optimized_prompt}
            json_data = {'prompt': optimized_prompt}
            
            # 打印调试信息
            print(f"提示词类型: {type(optimized_prompt)}")
            print(f"提示词值: {optimized_prompt}")
            
            # 发送请求
            print(f"发送请求到 {url}")
            print(f"图片文件: {os.path.basename(image_path)}")
            
            # 尝试同时使用多种参数传递prompt
            response = requests.post(
                url, 
                files=files, 
                data=data,
                params=params,
                json=json_data
            )
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                task_id = result.get('task_id')
                
                if task_id:
                    # 更新缓存
                    cache_entry["video_task_id"] = task_id
                    cache_entry["video_status"] = "processing"  # 更新视频状态
                    print(f"视频生成任务已提交，任务ID: {task_id}")
                    return task_id
                else:
                    print(f"未获取到有效的任务ID")
                    cache_entry["video_status"] = "failed"  # 更新视频状态为失败
                    return None
            else:
                print(f"视频生成API请求失败: {response.status_code}, {response.text}")
                cache_entry["video_status"] = "failed"  # 更新视频状态为失败
                return None
    except Exception as e:
        print(f"提交视频生成任务时出错: {str(e)}")
        if cache_entry:
            cache_entry["video_status"] = "failed"  # 更新视频状态为失败
        return None

def check_video_status(image_hash: str, cache: Dict[str, Any]) -> bool:
    """检查视频生成状态"""
    try:
        cache_entry = cache.get(image_hash)
        if not cache_entry:
            print(f"缓存中未找到图片 {image_hash}")
            return False
            
        # 如果已经有视频路径，表示已完成
        if cache_entry.get("video_path") and cache_entry.get("video_status") == "completed":
            print(f"视频已生成: {cache_entry['file_name']}")
            return True
            
        # 获取任务ID
        task_id = cache_entry.get("video_task_id")
        if not task_id:
            print(f"未找到视频任务ID: {cache_entry['file_name']}")
            return False
            
        # 获取输出目录
        output_dir = cache_entry.get("output_dir")
        if not output_dir:
            print(f"未找到输出目录: {cache_entry['file_name']}")
            return False
            
        # 检查任务状态
        url = f"{VIDEO_API_BASE}/task/{task_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            
            # 检查是否返回了Task not found
            if "detail" in result and result["detail"] == "Task not found":
                print(f"任务不存在: {task_id}")
                cache_entry["video_status"] = "failed"
                return True
                
            status = result.get('status')
            
            # 更新缓存中的视频状态
            cache_entry["video_status"] = status
            
            if status == "completed":
                # 视频生成完成
                video_result = result.get('result')
                
                if video_result:
                    # 提取文件名
                    filename = video_result.replace("output/", "")
                    
                    # 下载视频
                    download_url = f"{VIDEO_API_BASE}/video/{filename}"
                    video_response = requests.get(download_url, stream=True)
                    
                    if video_response.status_code == 200:
                        # 保存视频到输出目录
                        video_path = os.path.join(output_dir, filename)
                        with open(video_path, 'wb') as f:
                            for chunk in video_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # 更新缓存
                        cache_entry["video_path"] = video_path
                        
                        # 更新结果JSON
                        update_result_json(image_hash, cache)
                        
                        print(f"视频已下载保存: {video_path}")
                        return True
                    else:
                        print(f"下载视频失败: {video_response.status_code}, {video_response.text}")
                        cache_entry["video_status"] = "failed"  # 更新视频状态为失败
                        return False
                else:
                    print(f"视频生成完成但未找到结果: {result}")
                    cache_entry["video_status"] = "failed"  # 更新视频状态为失败
                    return False
            elif status == "error":
                print(f"视频生成失败: {result.get('error', '未知错误')}")
                cache_entry["video_status"] = "failed"  # 更新视频状态为失败
                return True  # 返回True表示任务已完成（虽然是错误状态）
            else:
                # 任务仍在处理中
                print(f"视频生成中: {cache_entry['file_name']}, 状态: {status}")
                return False
        else:
            response_text = response.text
            # 检查是否返回了Task not found
            try:
                error_data = response.json()
                if "detail" in error_data and error_data["detail"] == "Task not found":
                    print(f"任务不存在: {task_id}")
                    cache_entry["video_status"] = "failed"
                    return True
            except:
                pass
            
            print(f"检查视频状态失败: {response.status_code}, {response_text}")
            return False
    except Exception as e:
        print(f"检查视频状态时出错: {str(e)}")
        if cache_entry:
            cache_entry["video_status"] = "failed"  # 更新视频状态为失败
        return False

def update_result_json(image_hash: str, cache: Dict[str, Any]):
    """更新结果JSON文件"""
    try:
        if image_hash in cache and isinstance(cache[image_hash], dict):
            cache_entry = cache[image_hash]
            output_dir = cache_entry.get("output_dir")
            
            if output_dir and os.path.exists(output_dir):
                # 创建结果汇总文件
                result_summary = {
                    "file_name": cache_entry.get("file_name", ""),
                    "image_path": cache_entry.get("image_path", ""),
                    "optimized_prompt": cache_entry.get("optimized_prompt", ""),
                    "video_task_id": cache_entry.get("video_task_id", ""),
                    "video_path": cache_entry.get("video_path", ""),
                    "video_status": cache_entry.get("video_status", "")
                }
                
                summary_path = os.path.join(output_dir, "result.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(result_summary, f, ensure_ascii=False, indent=2)
                    
                # 复制图片到输出目录
                image_path = cache_entry.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    dest_image_path = os.path.join(output_dir, os.path.basename(image_path))
                    if not os.path.exists(dest_image_path):
                        shutil.copy2(image_path, dest_image_path)
                
                return True
    except Exception as e:
        print(f"更新结果JSON时出错: {str(e)}")
    return False

def check_pending_videos(cache: Dict[str, Any]) -> None:
    """检查所有待处理的视频生成任务，直到所有任务完成或失败"""
    pending_videos = []
    
    # 收集所有待处理或失败的视频任务
    for image_hash, entry in cache.items():
        if isinstance(entry, dict):
            # 处理中的任务
            if entry.get("video_task_id") and not entry.get("video_path"):
                pending_videos.append(image_hash)
            # 失败的任务，需要重试
            elif entry.get("video_status") == "failed" or entry.get("video_status") is None:
                if entry.get("optimized_prompt"):  # 确保有prompt可用
                    pending_videos.append(image_hash)
    
    if not pending_videos:
        print("没有待处理的视频任务")
        return
    
    print(f"发现 {len(pending_videos)} 个待处理或需要重试的视频任务")
    
    # 先尝试重新提交失败的任务
    for image_hash in pending_videos.copy():
        cache_entry = cache.get(image_hash)
        if cache_entry and (cache_entry.get("video_status") == "failed" or cache_entry.get("video_status") is None):
            # 清除旧的任务ID，准备重新提交
            if cache_entry.get("video_task_id"):
                print(f"重新提交失败的任务: {cache_entry.get('file_name')}")
                cache_entry["video_task_id"] = None
                cache_entry["video_path"] = None
            generate_video(image_hash, cache)
            # 保存缓存
            save_cache(cache)
    
    # 循环检查视频状态，直到所有任务都完成或失败
    print("开始持续监控所有视频任务状态...")
    
    while pending_videos:
        # 创建待移除的任务列表
        completed = []
        
        # 检查每个任务的状态
        for image_hash in pending_videos:
            cache_entry = cache.get(image_hash)
            # 检查是否已经是最终状态（完成或失败）
            if cache_entry and (cache_entry.get("video_status") == "completed" or cache_entry.get("video_status") == "failed"):
                completed.append(image_hash)
                print(f"任务已处理完成(状态: {cache_entry.get('video_status')}): {cache_entry.get('file_name')}")
                continue
                
            # 检查任务当前状态
            if check_video_status(image_hash, cache):
                cache_entry = cache.get(image_hash)  # 重新获取，因为check_video_status可能更新了状态
                # 再次检查是否为最终状态
                if cache_entry and (cache_entry.get("video_status") == "completed" or cache_entry.get("video_status") == "failed"):
                    completed.append(image_hash)
                    print(f"任务已处理完成(状态: {cache_entry.get('video_status')}): {cache_entry.get('file_name')}")
            
            # 保存缓存
            save_cache(cache)
        
        # 移除已完成或已失败的任务
        for image_hash in completed:
            pending_videos.remove(image_hash)
        
        # 如果还有待处理的任务，等待一段时间后再检查
        if pending_videos:
            print(f"仍有 {len(pending_videos)} 个视频任务处理中，等待 {VIDEO_CHECK_INTERVAL} 秒后继续检查...")
            time.sleep(VIDEO_CHECK_INTERVAL)
    
    print("所有视频任务都已达到最终状态(完成或失败)")

def cleanup_unnecessary_files(cache: Dict[str, Any]):
    """清理不需要的文件"""
    print("开始清理不必要的文件...")
    
    for image_hash, entry in cache.items():
        if isinstance(entry, dict):
            output_dir = entry.get("output_dir")
            if output_dir and os.path.exists(output_dir):
                # 删除不需要的JSON文件
                files_to_delete = [
                    os.path.join(output_dir, "image_analysis.json"),
                    os.path.join(output_dir, "optimized_prompt.json"),
                    os.path.join(output_dir, "prompt.json")
                ]
                
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"已删除: {file_path}")
                        except Exception as e:
                            print(f"删除文件失败: {file_path}, 错误: {str(e)}")
    
    print("文件清理完成")

def perform_image_processing(image_path: str, cache: Dict[str, Any]) -> bool:
    """执行完整的图片处理流程并返回是否需要检查视频状态"""
    try:
        image_hash = get_image_hash(image_path)
        file_name = os.path.basename(image_path)
        
        # 检查是否已完成处理
        if image_hash in cache and isinstance(cache[image_hash], dict):
            cache_entry = cache[image_hash]
            if cache_entry.get("video_path") and cache_entry.get("video_status") == "completed":
                print(f"图片已处理完成，跳过: {file_name}")
                return False
            
            # 如果视频状态为失败或为空，需要重新生成视频
            if cache_entry.get("video_status") == "failed" or cache_entry.get("video_status") is None:
                # 如果有优化后的提示词，直接重新生成视频
                if cache_entry.get("optimized_prompt"):
                    print(f"重新生成视频: {file_name}")
                    # 清除旧的任务ID
                    cache_entry["video_task_id"] = None
                    cache_entry["video_path"] = None
                    generate_video(image_hash, cache)
                    return True
        
        # 新图片或未完成的图片，执行完整流程
        print(f"处理新图片: {file_name}")
        
        # 步骤1: 分析图片
        analyze_image(image_path, cache)
        
        # 步骤2: 生成优化的prompt
        optimize_prompt(image_hash, cache)
        
        # 步骤3: 生成视频
        generate_video(image_hash, cache)
        
        # 保存缓存
        save_cache(cache)
        
        # 更新结果JSON
        update_result_json(image_hash, cache)
        
        return True  # 需要后续检查视频状态
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return False

def monitor_and_process():
    """监控数据集文件夹并处理新图片"""
    print("开始监控数据集文件夹并处理图片...")
    
    # 确保目录存在
    ensure_dirs()
    
    # 加载缓存
    cache = load_cache()
    print(f"已加载缓存，共 {len(cache)} 条记录")
    
    # 首先处理所有未完成的任务
    check_pending_videos(cache)
    
    while True:
        # 获取所有图片文件
        image_files = []
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith(SUPPORTED_FORMATS):
                    image_files.append(os.path.join(root, file))
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 标记是否有需要检查视频状态的任务
        need_check = False
        
        # 处理每张图片
        for img_path in image_files:
            if perform_image_processing(img_path, cache):
                need_check = True
            # 每处理一张图片后保存缓存
            save_cache(cache)
        
        # 检查视频生成状态
        if need_check:
            check_pending_videos(cache)
        
        # 清理不必要的文件
        cleanup_unnecessary_files(cache)
        
        # 等待一段时间后再检查
        print(f"等待60秒后再次检查数据集...")
        time.sleep(60)

def main():
    # 确保目录存在
    ensure_dirs()
    
    # 加载缓存
    cache = load_cache()
    print(f"已加载缓存，共 {len(cache)} 条记录")
    
    # 更新旧格式缓存项
    updated_count = 0
    for image_hash, entry in cache.items():
        # 对于字符串格式的简单缓存项，跳过更新
        if isinstance(entry, str):
            continue
            
        # 对于缺少输出目录的缓存项，更新它们
        if isinstance(entry, dict) and "output_dir" not in entry:
            file_name = entry.get("file_name")
            if file_name:
                entry["output_dir"] = ensure_image_dir(file_name)
                updated_count += 1
                
        # 对于缺少视频相关字段的缓存项，更新它们
        if isinstance(entry, dict) and "video_task_id" not in entry:
            entry["video_task_id"] = None
            entry["video_path"] = None
            updated_count += 1
    
    if updated_count > 0:
        print(f"已更新 {updated_count} 个缓存项以包含新的字段")
        save_cache(cache)
    
    # 清理不必要的文件
    cleanup_unnecessary_files(cache)
    
    # 开始监控和处理
    try:
        monitor_and_process()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序出错: {str(e)}")
    
    print("程序结束")

if __name__ == "__main__":
    main() 