#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/23 16:10
# @Author  : Oliver
# @File    : script_mk_gif.py
# @Software: PyCharm
import os
import re
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
from tqdm.auto import tqdm
from pathlib import Path
import re


def extract_date_time_from_string(input_string):
    """
    从输入字符串中提取日期和时间信息

    参数:
        input_string (str): 包含日期和时间的输入字符串

    返回:
        tuple: (日期字符串, 时间字符串)
               日期格式为YYYYMMDD，时间格式为HHMMSS
               如果未找到匹配，相应部分返回None
    """
    # 定义日期和时间模式
    date_pattern = r'(\d{8})(?:_|$)'
    time_pattern = r'_(\d{2}-\d{2}-\d{2})_'

    try:
        # 提取日期
        date_match = re.search(date_pattern, input_string)
        date_str = date_match.group(1) if date_match else None

        # 提取时间并去除分隔符
        time_match = re.search(time_pattern, input_string)
        time_str = time_match.group(1).replace('-', '') if time_match else None

        return date_str, time_str
    except Exception as e:
        print(f"提取日期或时间时出错: {e}")
        return None, None


def generate_animation(
        image_paths:list[str],
        # folder_path: str,
        output_path: str,
        cover_frame = 0,
        # pattern: str = r".*\.(jpg|jpeg|png|bmp|tiff)",
        output_format: str = "gif",
        fps: int = 10,
        duration: int = 100,
        resize: Optional[tuple] = None,
) -> None:
    """
    从文件夹中读取匹配正则表达式的图片序列，生成动态图（GIF 或视频）。

    Args:
        output_path: 输出动态图的路径（如 "output.gif" 或 "output.mp4"）。
        output_format: 输出格式，支持 "gif" 或 "mp4"。
        cover_frame: 将第几帧作为封面额外存储为一个重命名为cover_(output_path.filename).png图片。
        fps: 视频帧率（仅用于 MP4）。
        duration: GIF 每帧持续时间（毫秒）。
        resize: 调整图片大小（如 (width, height)），None 表示不调整。
    """
    if cover_frame >=len(image_paths) or cover_frame<0:
        cover_frame = 0

    # Handle cover frame
    cover_img = cv2.imread(image_paths[cover_frame])
    if resize:
        cover_img = cv2.resize(cover_img, resize)

    # Generate cover filename
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    cover_filename = f"cover_{output_filename.split('.')[0]}.png"
    cover_path = os.path.join(output_dir, cover_filename)

    # Save cover image
    cv2.imwrite(cover_path, cover_img)
    print(f"Cover frame saved to {cover_path}")


    first_image_path = image_paths[0]
    first_image = cv2.imread(first_image_path)
    if resize:
        first_image = cv2.resize(first_image, resize)
    height, width = first_image.shape[:2]

    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Existing file {output_path} has been removed.")

    # 3. 生成动态图
    if output_format.lower() == "gif":
        # 生成 GIF
        images = []
        # for image_path in image_paths:
        for image_path in tqdm(image_paths, desc="Processing images for GIF"):
            # image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            if resize:
                img = cv2.resize(img, resize)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 是 BGR，PIL 需要 RGB
            images.append(Image.fromarray(img))

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF saved to {output_path}")

    elif output_format.lower() == "mp4":
        # 生成 MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        count = 0
        for image_path in tqdm(image_paths, desc="Processing images for mp4"):
        # for image_path in image_paths:
            # image_path = os.path.join(folder_path, image_file)
            img = cv2.imread(image_path)
            if resize:
                img = cv2.resize(img, resize)
            video.write(img)
            count += 1
        video.release()
        print(f"MP4 saved to {output_path}, frames: {count}")

    else:
        raise ValueError("Unsupported output format. Use 'gif' or 'mp4'.")

pattern_dict = {"Combine_diffusion":r"combined_diffusion_analysis_diffusion_combine_([0-9]*)\.png",
                "Combine_pearson":r"combined_pearson_fd_([0-9]*)\.png",

                "Defuzzifier":r"explain_2d_defuzzifier_defuzzifier_([0-9]*)\.png",
                "Defuzzifier-16":r"explain_2d_defuzzifier_len\(16\)_defuzzifier_([0-9]*)\.png",
                "Defuzzifier-64":r"explain_2d_defuzzifier_len\(64\)_defuzzifier_([0-9]*)\.png",

                "Ante": r"explain_2d_samples_([0-9]*)\.png",
                "Ante_all": r"explain_2d_samples_([0-9]*)_2\.png",

                # "Displace_distribution":r"diffusion_analyze_([0-9]*)_displace_distribution\.png",
                # "Track_plot":r"diffusion_analyze_([0-9]*)_track_plot\.png",
                # "Correlate_heatmap":r"correlation_heatmap_mpl_([0-9]*)\.png",
                # "Raw_MSD-t":r"diffusion_MSD_diffusion_analyze_([0-9]*)\.png",
                # "log_MSD-t":r"diffusion_MSD_ln_diffusion_analyze_([0-9]*)\.png",
                # "Correlate_force_direction":r"FD_([0-9]*)\.png",
                "Firing_level":r"interpretable_frequency_([0-9]*)_fl_sort\.png",
                }
# source_dir = r"output/runs_log/20250725_19-26-51_Test_CompareAnte_dataset_quin33_20s_Linux/pics"  # old default
# flag_norm = False
# source_dir = r"output/runs_log/20250725_18-31-19_Test_CompareAnte_dataset_quin33_20s_Linux/pics"    # old norm
# flag_norm = True


# source_dir = r"output/runs_log/20250730_11-17-01_Test_CompareAnte_dataset_quin33_20s_Linux/pics"  # default
# flag_norm = False
source_dir = r"output/runs_log/20250730_10-26-12_Test_CompareAnte_dataset_quin33_20s_Linux/pics"  # norm
flag_norm = True

describe = "-normed" if flag_norm else ""
gif_save_dir = "gif_result"
save_prefix = "-".join(extract_date_time_from_string(source_dir)) + describe
cover = 150
if __name__ == "__main__":
    for output_name, file_pattern in pattern_dict.items():
        print(f"Generating animation for [{output_name}]")
        # 从 "images" 文件夹读取所有 "frame_*.png" 文件，生成 GIF
        image_files = [
            f for f in os.listdir(source_dir)
            if re.match(file_pattern, f, re.IGNORECASE)
        ]
        sorted_files = sorted(image_files, key=lambda f: int(re.search(file_pattern, f).group(1)))
        if len(sorted_files) == 0:
            print(f"No images found in {source_dir} for {output_name}")
            continue
        else:
            print(f"Found {len(sorted_files)} images in {source_dir}: ")
            if not os.path.exists(os.path.join(gif_save_dir, save_prefix)):
                os.makedirs(os.path.join(gif_save_dir, save_prefix))

        image_paths = [os.path.join(source_dir,i) for i in sorted_files]

        generate_animation(
            image_paths=image_paths,
            output_path=os.path.join(gif_save_dir,save_prefix, output_name+".gif"),
            output_format="gif",
            duration=41,  # 每帧 100ms
            cover_frame = cover,
            # resize=(640, 480),  # 调整大小
        )

        # 生成 MP4
        generate_animation(
            image_paths=image_paths,
            output_path=os.path.join(gif_save_dir, save_prefix, output_name+".mp4"),
            output_format="mp4",
            fps=6,  # 24 帧每秒
            cover_frame = cover,
            # resize=(680, 420),  # 调整大小
        )
