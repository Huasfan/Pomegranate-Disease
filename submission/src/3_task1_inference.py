import os
import glob
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# --- 配置 ---
IMAGE_DIR = r"datasets/all_images"
MODEL_PATH = "mods/unet_student_semi.pth"  # 使用半监督模型
OUTPUT_FILE = "result1.xlsx"

# 阶段划分阈值 (病斑占比 P)
THRESH_HEALTHY = 0.005  # 0.5%
THRESH_INITIAL = 0.050  # 5.0%


# --- 模型结构  ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.ups = nn.ModuleList();
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        features = [64, 128, 256, 512]

        for f in features:
            self.downs.append(DoubleConv(in_channels, f));
            in_channels = f
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2));
            self.ups.append(DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs: x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x);
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x);
            skip = skips[idx // 2]
            if x.shape != skip.shape: x = TF.resize(x, skip.shape[2:])
            x = self.ups[idx + 1](torch.cat((skip, x), 1))
        return self.final_conv(x)


def get_fruit_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    if len(parts) >= 3 and parts[0].startswith('w'):
        return f"week_{parts[0][1:]}", f"{int(parts[1][1:])}.{int(parts[2][1:])}"
    return None, None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUNet().to(device)

    if os.path.exists(MODEL_PATH):
        # weights_only=False 兼容旧版保存方式
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("模型加载成功！")
    else:
        print(f"模型文件 {MODEL_PATH} 未找到！")
        return

    model.eval()

    results = []
    images = [f for f in glob.glob(os.path.join(IMAGE_DIR, "*.*")) if f.lower().endswith(('.jpg', '.png'))]
    print(f"正在推理{len(images)} 张图片...")

    with torch.no_grad():
        for i, path in enumerate(images):
            img = cv2.imread(path)
            if img is None: continue
            h, w = img.shape[:2]

            x = TF.resize(TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0), [256, 256]).to(device)

            # 预测 Mask
            logits = model(x)
            pred_mask = TF.resize((torch.sigmoid(logits) > 0.5).float(), [h, w]).squeeze().cpu().numpy()

            # 计算占比
            disease_px = np.count_nonzero(pred_mask)
            fruit_px = np.count_nonzero(get_fruit_mask(img))
            if fruit_px < 100: fruit_px = h * w  # 容错

            ratio = disease_px / fruit_px
            stage = "健康期" if ratio <= THRESH_HEALTHY else ("初发期" if ratio <= THRESH_INITIAL else "发病期")

            wk, fid = parse_filename(os.path.basename(path))
            if wk: results.append({"周数": wk, "果实编号": fid, "果实阶段": stage})

            if (i + 1) % 100 == 0: print(f"已推理{i + 1}...")

    pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
    print(f"结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()