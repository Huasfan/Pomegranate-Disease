import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import glob

# ================= 配置区域 =================
IMG_PATH = r"datasets/all_images/w09_t07_f17.jpg" #图片路径
MODEL_PATH = "mods/unet_student_semi.pth"
SAVE_PATH = "unet_heatmap.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= 模型定义  =================
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


def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = SimpleUNet().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return
    else:
        print(f"模型文件 {MODEL_PATH} 未找到")
        return

    model.eval()


    target_img_path = IMG_PATH

    # 检查图片是否存在，不存在则自动搜索
    if not os.path.exists(target_img_path):
        print(f"默认图片未找到: {target_img_path}")
        # 尝试自动找一张图
        possible_imgs = glob.glob(os.path.join("datasets", "all_images", "*.jpg"))
        if possible_imgs:
            target_img_path = possible_imgs[0]
            print(f"已自动切换到测试图片: {target_img_path}")
        else:
            print("all_images目录下没有找到.jpg图片")
            return

    # 读取图片
    original_img = cv2.imread(target_img_path)
    if original_img is None:
        print(f"无法读取图片文件: {target_img_path}")
        return

    h, w = original_img.shape[:2]
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # 预处理
    x = TF.to_tensor(img_rgb)
    x = TF.resize(x, [256, 256]).unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    # 后处理可视化
    prob_map = TF.resize(probs, [h, w]).squeeze().cpu().numpy()

    # 制作热力图
    heatmap_uint8 = (prob_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 叠加
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

    # 拼接
    combined = np.hstack([original_img, overlay])
    cv2.imwrite(SAVE_PATH, combined)

    print(f"可视化完成！结果已保存为: {SAVE_PATH}")
    print(f"处理图片: {target_img_path}")


if __name__ == "__main__":
    main()