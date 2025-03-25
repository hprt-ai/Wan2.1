#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import time
from datetime import datetime
import image_analysis

# 配置日志
def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"image_analysis_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("image_processor")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图片分析和提示词生成系统")
    parser.add_argument("--input", "-i", type=str, default="dataset", 
                        help="输入图片目录 (默认: dataset)")
    parser.add_argument("--output", "-o", type=str, default="output", 
                        help="输出结果目录 (默认: output)")
    parser.add_argument("--workers", "-w", type=int, default=1, 
                        help="并行处理的线程数 (默认: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="显示详细日志")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="强制重新处理所有图片")
    return parser.parse_args()

def check_environment():
    """检查环境变量和依赖"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误: 环境变量 DASHSCOPE_API_KEY 未设置")
        print("请设置环境变量: export DASHSCOPE_API_KEY=your_api_key")
        sys.exit(1)

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    # 检查环境
    check_environment()
    
    # 更新配置
    image_analysis.INPUT_DIR = args.input
    image_analysis.OUTPUT_DIR = args.output
    image_analysis.MAX_WORKERS = args.workers
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"开始处理图片，输入目录: {args.input}, 输出目录: {args.output}")
    
    try:
        # 如果指定了强制重新处理，则删除缓存文件
        if args.force and os.path.exists(image_analysis.CACHE_FILE):
            logger.info("强制模式: 删除缓存文件")
            os.remove(image_analysis.CACHE_FILE)
        
        # 运行图片分析
        image_analysis.main()
        
        # 记录完成时间
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，用时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 