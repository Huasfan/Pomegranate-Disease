import os
import json
import numpy as np
import cv2
import glob

# ================= 配置区域 =================
JSON_DIR = r"datasets/task1_train/jsons"         # JSON 标签文件夹
OUTPUT_MASK_DIR = r"datasets/task1_train/masks"  # Mask 输出文件夹


def generate_masks():
    if not os.path.exists(OUTPUT_MASK_DIR):
        os.makedirs(OUTPUT_MASK_DIR)
        print(f"创建输出目录: {OUTPUT_MASK_DIR}")

    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"正在处理 {len(json_files)} 个 JSON 文件...")

    count = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            h = data.get("imageHeight")
            w = data.get("imageWidth")

            if h is None or w is None:
                img_name = data.get("imagePath")
                # 尝试猜测图片路径
                img_path_guess = os.path.join(os.path.dirname(JSON_DIR), "images", img_name)
                if os.path.exists(img_path_guess):
                    temp = cv2.imread(img_path_guess)
                    if temp is not None:
                        h, w = temp.shape[:2]

                # 如果还是找不到尺寸，跳过该文件
                if h is None:
                    print(f"无法获取尺寸，跳过: {os.path.basename(json_path)}")
                    continue

            # 创建全黑 Mask (背景)
            mask = np.zeros((h, w), dtype=np.uint8)

            # 遍历所有标注形状
            shapes = data.get("shapes", [])
            for shape in shapes:
                points = np.array(shape.get("points"), dtype=np.int32)
                shape_type = shape.get("shape_type", "polygon")

                # 将所有标注区域填充为白色 (255)
                if shape_type == "polygon" or shape_type == "linestrip":
                    cv2.fillPoly(mask, [points], 255)
                elif shape_type == "rectangle":
                    cv2.rectangle(mask, tuple(points[0]), tuple(points[1]), 255, -1)
                elif shape_type == "circle":
                    center = tuple(points[0])
                    radius = int(np.linalg.norm(points[0] - points[1]))
                    cv2.circle(mask, center, radius, 255, -1)

            # 保存 Mask
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            output_path = os.path.join(OUTPUT_MASK_DIR, base_name + ".png")
            cv2.imwrite(output_path, mask)
            count += 1

        except Exception as e:
            print(f"处理出错 {json_path}: {e}")

    print(f"共生成 {count} 张掩膜。")
    print("现在可以运行 2_task1_train_semi.py 进行训练了。")


if __name__ == "__main__":
    generate_masks()