import os
import glob
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# --- 配置 ---
IMAGE_DIR = r"datasets/all_images"
MODEL_PATH = "unet_mech_student.pth"
OUTPUT_FILE = "result4.xlsx"
# 像素阈值过滤：用于排除噪点(<5)和误判的大面积病斑(>5000)
MIN_PIXELS, MAX_PIXELS = 5, 5000


# --- 模型结构  ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.ups = nn.ModuleList();
        self.downs = nn.ModuleList();
        self.pool = nn.MaxPool2d(2)

        feats = [64, 128, 256, 512]

        for f in feats: self.downs.append(DoubleConv(in_c, f)); in_c = f
        for f in reversed(feats): self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2)); self.ups.append(
            DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2);
        self.final = nn.Conv2d(feats[0], out_c, 1)

    def forward(self, x):
        skips = [];
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x);
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x);
            s = skips[i // 2];
            if x.shape != s.shape: x = TF.resize(x, s.shape[2:])
            x = self.ups[i + 1](torch.cat((s, x), 1))
        return self.final(x)


def parse_fn(fn):
#解析文件名 wXX_tXX_fXX.jpg 提取周数和编号
    p = os.path.splitext(fn)[0].split('_')
    if len(p) >= 3 and p[0].startswith('w'):
        return f"week_{p[0][1:]}", f"{int(p[1][1:])}_{int(p[2][1:])}"
    return None, None


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Inference on {dev}...")

    # 1. 加载模型
    model = SimpleUNet().to(dev)
    if os.path.exists(MODEL_PATH):
        # weights_only=False
        model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
        print(f"模型加载完成: {MODEL_PATH}")
    else:
        print(f"Error: 未找到权重模型 {MODEL_PATH}")
        return
    model.eval()

    res = []
    files = glob.glob(os.path.join(IMAGE_DIR, "*.*"))

    # 2. 批量推理
    with torch.no_grad():
        for i, path in enumerate(files):
            if not path.lower().endswith(('.jpg', '.png')): continue

            img = cv2.imread(path)
            if img is None: continue

            # 预处理：BGR转RGB -> Tensor -> Resize (256x256)
            x = TF.resize(TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0), [256, 256]).to(dev)

            # 推理并统计像素数
            mask_pred = torch.sigmoid(model(x)) > 0.5
            px_count = torch.sum(mask_pred).item()

            # 3. 只保留符合机械损伤特征范围的样本
            if MIN_PIXELS < px_count < MAX_PIXELS:
                w, f = parse_fn(os.path.basename(path))
                if w: res.append({"周数": w, "果实编号": f})

            if (i + 1) % 500 == 0: print(f"已处理 {i + 1} 张图像...")

    # 4. 保存结果
    pd.DataFrame(res).to_excel(OUTPUT_FILE, index=False)
    print(f"Task 4推理完成. 找到 {len(res)} 机械损伤样本，已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()