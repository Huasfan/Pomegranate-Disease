import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm

# 附件1文件夹路径
SOURCE_DIR = r"附件1"
# 目标文件夹
TARGET_DIR = r"datasets/all_images"


# ===========================================

def flatten_dataset():
    # 1. 创建目标文件夹
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"创建目标目录: {TARGET_DIR}")
    else:
        print(f"目标目录已存在: {TARGET_DIR} (新图片将追加到此目录)")

    print(f"开始扫描 '{SOURCE_DIR}' ...")

    copy_count = 0
    error_count = 0

    # 遍历所有文件
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            # 过滤非图片文件
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            src_path = os.path.join(root, file)

            try:
                # 将路径拆分为各部分
                path_parts = Path(src_path).parts

                week_num = None
                for part in path_parts:
                    match = re.search(r'week_?(\d+)', part, re.IGNORECASE)
                    if match:
                        week_num = int(match.group(1))
                        break

                if week_num is None:
                    continue

                # 2. 获取树号和果实号
                file_name_no_ext = os.path.splitext(file)[0]

                if '-' in file_name_no_ext:
                    tree_str, fruit_str = file_name_no_ext.split('-')
                elif '_' in file_name_no_ext:
                    tree_str, fruit_str = file_name_no_ext.split('_')
                else:
                    # 无法解析的文件名格式
                    print(f"跳过未知格式文件: {file}")
                    error_count += 1
                    continue

                tree_id = int(tree_str)
                fruit_id = int(fruit_str)

                # 3. 构建新文件名: wXX_tXX_fXX.jpg
                # w14_t09_f01.jpg
                new_name = f"w{week_num:02d}_t{tree_id:02d}_f{fruit_id:02d}.jpg"
                dst_path = os.path.join(TARGET_DIR, new_name)

                # 4. 复制文件
                shutil.copy2(src_path, dst_path)
                copy_count += 1

                if copy_count % 100 == 0:
                    print(f" 已处理 {copy_count} 张: {new_name}")

            except Exception as e:
                print(f"处理出错 {src_path}: {e}")
                error_count += 1

    print("-" * 30)
    print(f"数据扁平化完成")
    print(f"  - 成功复制: {copy_count} 张")
    print(f"  - 失败/跳过: {error_count} 张")
    print(f"  - 图片保存位置: {os.path.abspath(TARGET_DIR)}")


if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"错误: 找不到源文件夹 '{SOURCE_DIR}'。")
    else:
        flatten_dataset()