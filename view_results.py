#!/usr/bin/env python3
import os
import json
import argparse
from tabulate import tabulate
from pathlib import Path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="查看图片分析和提示词生成结果")
    parser.add_argument("--cache", "-c", type=str, default="output/cache.json", 
                        help="缓存文件路径 (默认: output/cache.json)")
    parser.add_argument("--format", "-f", type=str, choices=["table", "json", "csv"], default="table", 
                        help="输出格式 (默认: table)")
    parser.add_argument("--output", "-o", type=str, 
                        help="输出文件路径（默认输出到控制台）")
    parser.add_argument("--limit", "-l", type=int, default=0, 
                        help="限制显示的条目数量 (默认: 0, 显示全部)")
    parser.add_argument("--filter", type=str, 
                        help="按文件名筛选结果 (支持部分匹配)")
    return parser.parse_args()

def load_cache(cache_file):
    """加载缓存文件"""
    if not os.path.exists(cache_file):
        print(f"错误: 缓存文件 {cache_file} 不存在")
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 缓存文件 {cache_file} 格式不正确")
        return {}

def prepare_display_data(cache, filter_term=None):
    """准备显示数据"""
    display_data = []
    
    for image_hash, entry in cache.items():
        # 跳过旧格式的缓存项
        if isinstance(entry, str):
            continue
            
        # 如果不是字典格式，跳过
        if not isinstance(entry, dict):
            continue
            
        file_name = entry.get("file_name", "未知")
        
        # 应用过滤器
        if filter_term and filter_term.lower() not in file_name.lower():
            continue
            
        image_path = entry.get("image_path", "未知")
        image_recognition = entry.get("image_recognition", "未生成")
        if image_recognition and len(image_recognition) > 50:
            image_recognition = image_recognition[:50] + "..."
            
        image_prompt = entry.get("image_prompt", "未生成")
        optimized_prompt = entry.get("optimized_prompt", "未生成")
        output_dir = entry.get("output_dir", "未创建")
        
        display_data.append({
            "hash": image_hash[:8] + "...",
            "文件名": Path(file_name).name,
            "图片路径": Path(image_path).name if image_path != "未知" else "未知",
            "输出目录": Path(output_dir).name if output_dir != "未创建" else "未创建",
            "图像识别": image_recognition,
            "初始提示词": image_prompt,
            "优化提示词": optimized_prompt
        })
    
    return display_data

def output_table(data, output_file=None):
    """以表格形式输出数据"""
    if not data:
        print("没有找到符合条件的数据")
        return
        
    table = tabulate(data, headers="keys", tablefmt="grid")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"表格已保存到 {output_file}")
    else:
        print(table)

def output_json(data, output_file=None):
    """以JSON格式输出数据"""
    if not data:
        print("没有找到符合条件的数据")
        return
        
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON已保存到 {output_file}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

def output_csv(data, output_file=None):
    """以CSV格式输出数据"""
    if not data:
        print("没有找到符合条件的数据")
        return
    
    headers = data[0].keys()
    csv_data = ",".join(headers) + "\n"
    
    for row in data:
        csv_row = []
        for h in headers:
            value = str(row[h]).replace('"', '""')
            csv_row.append(f'"{value}"')
        csv_data += ",".join(csv_row) + "\n"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(csv_data)
        print(f"CSV已保存到 {output_file}")
    else:
        print(csv_data)

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 加载缓存
    cache = load_cache(args.cache)
    
    if not cache:
        print("缓存为空或加载失败")
        return
    
    # 准备显示数据
    display_data = prepare_display_data(cache, args.filter)
    
    # 应用限制
    if args.limit > 0 and len(display_data) > args.limit:
        display_data = display_data[:args.limit]
    
    # 输出结果
    if args.format == "table":
        try:
            output_table(display_data, args.output)
        except ImportError:
            print("警告: tabulate模块未安装，使用JSON格式输出")
            output_json(display_data, args.output)
    elif args.format == "json":
        output_json(display_data, args.output)
    elif args.format == "csv":
        output_csv(display_data, args.output)

if __name__ == "__main__":
    main() 