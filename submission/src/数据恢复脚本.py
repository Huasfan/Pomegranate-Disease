import os
import shutil
from pathlib import Path


def copy_matched_images():
    all_images_dir = Path("datasets/all_images")
    datasets_dir = Path("datasets")
    task_dirs = ["task1_train", "task4_train"]

    if not all_images_dir.exists():
        print(f"错误：all_images文件夹不存在 -> {all_images_dir.absolute()}")
        return

    if not datasets_dir.exists():
        print(f"错误：datasets文件夹不存在 -> {datasets_dir.absolute()}")
        return

    for task_dir in task_dirs:
        task_path = datasets_dir / task_dir
        jsons_dir = task_path / "jsons"
        target_images_dir = task_path / "images"


        if not task_path.exists():
            print(f"警告：任务文件夹不存在，跳过 -> {task_path.absolute()}")
            continue
        if not jsons_dir.exists():
            print(f"警告：{task_dir}下无jsons文件夹，跳过 -> {jsons_dir.absolute()}")
            continue

        target_images_dir.mkdir(exist_ok=True)

        json_names = set()
        for json_file in jsons_dir.glob("*.json"):
            if json_file.is_file():
                json_names.add(json_file.stem)

        if not json_names:
            print(f"警告：{task_dir}/jsons中无json文件，跳过")
            continue

        copied_count = 0
        image_extensions = (".jpg", ".jpeg", ".png")
        for image_file in all_images_dir.glob("*.*"):
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                if image_file.stem in json_names:
                    target_path = target_images_dir / image_file.name
                    if not target_path.exists():
                        shutil.copy2(image_file, target_path)
                        copied_count += 1
                    # else:
                    #     print(f"已存在，跳过 -> {image_file.name}")

        print(f"{task_dir}：成功复制 {copied_count} 张图片到 {target_images_dir.absolute()}")

    print("\n所有任务处理完成")


if __name__ == "__main__":
    copy_matched_images()