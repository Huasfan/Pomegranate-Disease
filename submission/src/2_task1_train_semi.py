import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
import random
import shutil

# ================= 配置区域 =================
# 训练数据路径
REAL_IMG_DIR = r"datasets/task1_train/images"
REAL_MASK_DIR = r"datasets/task1_train/masks"
UNLABELED_IMG_DIR = r"datasets/all_images"

# 伪标签输出路径
PSEUDO_ROOT = r"datasets/task1_pseudo_set"
PSEUDO_IMG_DIR = os.path.join(PSEUDO_ROOT, "images")
PSEUDO_MASK_DIR = os.path.join(PSEUDO_ROOT, "masks")

TEACHER_MODEL_PATH = "mods/unet_teacher.pth"
STUDENT_MODEL_PATH = "mods/unet_student_semi.pth"

# 训练参数
BATCH_SIZE = 24
LR = 1e-4
TEACHER_EPOCHS = 40
STUDENT_EPOCHS = 40
CONFIDENCE_THRESHOLD = 0.80  # 伪标签置信度阈值
MAX_PSEUDO_SAMPLES = 500  # 最大伪标签样本数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 组合损失函数 (Dice Loss+BCE Loss)
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # BCE Loss
        bce = torch.nn.functional.binary_cross_entropy(inputs, targets)

        # 加权组合
        return 0.5 * bce + 0.5 * dice_loss


# ================= 模型定义 (UNet) =================
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


# ================= 数据集定义 =================
class PomegranateDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 仅读取图片文件
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 寻找对应的 Mask
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base_name + ".jpg")

        img = cv2.imread(img_path)
        if img is None:
            # 容错：若图片损坏，返回全黑
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取 Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            if mask is None: mask = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        mask[mask > 0] = 1.0

        if self.transform:
            img = TF.to_tensor(img)
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            if random.random() > 0.5: img = TF.hflip(img); mask = TF.hflip(mask)
            if random.random() > 0.5: img = TF.vflip(img); mask = TF.vflip(mask)
            img = TF.resize(img, [256, 256])
            mask = TF.resize(mask, [256, 256])

        return img, mask


# --- 训练核心函数 ---
def train_one_epoch(model, loader, optimizer, loss_fn, epoch_idx, total_epochs):
    model.train()
    loop_loss = 0
    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        predictions = model(data)

        if targets.dim() == 4 and targets.shape[1] == 1: targets = targets.squeeze(1)
        if predictions.dim() == 4 and predictions.shape[1] == 1: predictions = predictions.squeeze(1)

        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()
        loop_loss += loss.item()
    print(f"Epoch [{epoch_idx + 1}/{total_epochs}] Avg Loss: {loop_loss / len(loader):.4f}")


def generate_pseudo_labels(teacher_model):
    print("\n正在挖掘伪标签 (Pseudo-Labeling)...")
    if os.path.exists(PSEUDO_ROOT): shutil.rmtree(PSEUDO_ROOT)
    os.makedirs(PSEUDO_IMG_DIR);
    os.makedirs(PSEUDO_MASK_DIR)

    real_imgs = set(os.listdir(REAL_IMG_DIR))
    all_files = glob.glob(os.path.join(UNLABELED_IMG_DIR, "*.*"))

    # 仅筛选图片文件
    candidates = [
        p for p in all_files
        if os.path.basename(p) not in real_imgs
           and p.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    print(f"扫描到 {len(candidates)} 张候选图片，正在筛选...")

    teacher_model.eval()
    count = 0
    with torch.no_grad():
        for path in candidates:
            if count >= MAX_PSEUDO_SAMPLES: break
            img = cv2.imread(path)
            if img is None: continue
            h, w = img.shape[:2]
            x = TF.resize(TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0), [256, 256]).to(DEVICE)

            logits = teacher_model(x)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()

            # 计算置信度
            fg_conf = probs[pred_mask == 1].mean().item() if (pred_mask == 1).any() else 0
            bg_conf = (1 - probs)[pred_mask == 0].mean().item() if (pred_mask == 0).any() else 0
            confidence = fg_conf if (pred_mask == 1).any() else bg_conf

            if confidence > CONFIDENCE_THRESHOLD:
                shutil.copy(path, os.path.join(PSEUDO_IMG_DIR, os.path.basename(path)))
                mask_np = TF.resize(pred_mask, [h, w],
                                    interpolation=TF.InterpolationMode.NEAREST).squeeze().cpu().numpy() * 255

                save_name = os.path.splitext(os.path.basename(path))[0] + ".png"
                cv2.imwrite(os.path.join(PSEUDO_MASK_DIR, save_name), mask_np.astype(np.uint8))
                count += 1
                if count % 50 == 0: print(f"   已生成 {count} 个伪标签...")

    print(f"挖掘完成，共生成 {count} 对伪标签数据。")
    return count


def main():
    print(f"启动半监督分割训练 (Device: {DEVICE})")

    # --- Step 1: 训练教师模型 ---
    print(f"\n=== Phase 1: 训练 Teacher Model ({TEACHER_EPOCHS} epochs) ===")
    teacher = SimpleUNet().to(DEVICE)
    opt = optim.Adam(teacher.parameters(), lr=LR)
    loss_fn = CombinedLoss()

    ds_real = PomegranateDataset(REAL_IMG_DIR, REAL_MASK_DIR, transform=True)
    # num_workers=0 避免 Windows 下多进程文件锁问题
    dl_real = DataLoader(ds_real, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for epoch in range(TEACHER_EPOCHS):
        train_one_epoch(teacher, dl_real, opt, loss_fn, epoch, TEACHER_EPOCHS)

    torch.save(teacher.state_dict(), TEACHER_MODEL_PATH)
    print("教师模型训练完毕。")

    # --- Step 2: 生成伪标签 ---
    n_pseudo = generate_pseudo_labels(teacher)

    if n_pseudo == 0:
        print("未生成任何伪标签，跳过半监督阶段，直接保存教师模型作为最终模型。")
        shutil.copy(TEACHER_MODEL_PATH, STUDENT_MODEL_PATH)
        return

    # --- Step 3: 训练学生模型 ---
    print(f"\n=== Phase 2: 训练 Student Model ({STUDENT_EPOCHS} epochs) ===")
    ds_pseudo = PomegranateDataset(PSEUDO_IMG_DIR, PSEUDO_MASK_DIR, transform=True)
    ds_combined = ConcatDataset([ds_real, ds_pseudo])
    dl_combined = DataLoader(ds_combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    student = SimpleUNet().to(DEVICE)
    student.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))

    opt_student = optim.Adam(student.parameters(), lr=LR)

    for epoch in range(STUDENT_EPOCHS):
        train_one_epoch(student, dl_combined, opt_student, loss_fn, epoch, STUDENT_EPOCHS)

    torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    print(f"\n半监督训练全部完成！最终模型已保存为: {STUDENT_MODEL_PATH}")


if __name__ == "__main__":
    main()