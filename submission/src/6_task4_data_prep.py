import os
import json
import numpy as np
import cv2
import glob

# --- 配置 ---
JSON_DIR = r"datasets/task4_train/jsons"
OUTPUT_DIR = r"datasets/task4_train/masks"
TARGET_LABEL = "机械损伤"


def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 遍历所有JSON文件
    for jp in glob.glob(os.path.join(JSON_DIR, "*.json")):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)

            h, w = d.get("imageHeight"), d.get("imageWidth")
            if not h or not w: continue

            # 创建全黑背景
            mask = np.zeros((h, w), dtype=np.uint8)

            for s in d.get("shapes", []):
                # 精确匹配标签
                if s.get("label") == TARGET_LABEL:
                    pts = np.array(s.get("points"), dtype=np.int32)
                    stype = s.get("shape_type")

                    if stype in ["polygon", "linestrip"]:
                        cv2.fillPoly(mask, [pts], 255)
                    elif stype == "rectangle":
                        cv2.rectangle(mask, tuple(pts[0]), tuple(pts[1]), 255, -1)
                    elif stype == "circle":
                        rad = int(np.linalg.norm(pts[0] - pts[1]))
                        cv2.circle(mask, tuple(pts[0]), rad, 255, -1)

            # 保存Mask
            save_path = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(jp))[0] + ".png")
            cv2.imwrite(save_path, mask)

        except Exception:
            pass


if __name__ == "__main__":
    main()