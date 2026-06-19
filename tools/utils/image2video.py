import argparse
import cv2
import os
import glob
from pathlib import Path

def images_to_video(image_folder, output_video, fps=30, extension='jpg', resize=None):
    """
    将图片文件夹转换为视频
    
    Args:
        image_folder: 图片文件夹路径
        output_video: 输出视频文件路径
        fps: 帧率
        extension: 图片扩展名
        resize: 调整尺寸 (width, height)
    """
    
    # 检查输入文件夹
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"文件夹不存在: {image_folder}")
    
    # 支持的图片格式
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    if extension.lower() not in extensions:
        extensions.append(extension.lower())
    
    # 获取所有图片文件
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext}')))
        image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext.upper()}')))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"在 {image_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        raise ValueError("无法读取第一张图片")
    
    if resize:
        width, height = resize
        first_image = cv2.resize(first_image, (width, height))
    else:
        height, width = first_image.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 处理所有图片
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is not None:
            if resize:
                img = cv2.resize(img, (width, height))
            video.write(img)
            print(f"处理进度: {i+1}/{len(image_paths)}", end='\r')
        else:
            print(f"警告: 无法读取图片 {image_path}")
    
    # 释放资源
    video.release()
    print(f"\n视频已保存到: {output_video}")

def main():
    parser = argparse.ArgumentParser(description='将图片文件夹转换为视频')
    parser.add_argument('input_folder', help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', default='output.mp4', help='输出视频文件名')
    parser.add_argument('-f', '--fps', type=int, default=30, help='帧率 (默认: 30)')
    parser.add_argument('-e', '--extension', default='jpg', help='图片扩展名 (默认: jpg)')
    parser.add_argument('-r', '--resize', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='调整视频尺寸')
    
    args = parser.parse_args()
    
    try:
        images_to_video(
            args.input_folder,
            args.output,
            args.fps,
            args.extension,
            args.resize
        )
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    # 直接运行示例
    # images_to_video("images_folder", "output.mp4", fps=30)
    
    # 或运行命令行工具
    main()