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
from pathlib import Path

# 配置
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REAL_IMG_DIR = os.path.join(BASE_DIR, "datasets", "train4_train", "images")
REAL_MASK_DIR = os.path.join(BASE_DIR, "datasets", "train4_train", "masks")
UNLABELED_IMG_DIR = os.path.join(BASE_DIR, "datasets", "all_images")

PSEUDO_ROOT = os.path.join(BASE_DIR, "datasets", "task4_pseudo_set")
PSEUDO_IMG_DIR = os.path.join(PSEUDO_ROOT, "images")
PSEUDO_MASK_DIR = os.path.join(PSEUDO_ROOT, "masks")

TEACHER_MODEL_PATH = "mods/unet_mech_teacher.pth"
STUDENT_MODEL_PATH = "mods/unet_mech_student.pth"

BATCH_SIZE = 8
LR = 1e-4
TEACHER_EPOCHS = 40
STUDENT_EPOCHS = 40
CONFIDENCE_THRESHOLD = 0.75
MAX_PSEUDO_SAMPLES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 加权损失函数
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.weight = torch.tensor([50.0]).to(DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.weight)

    def forward(self, inputs, targets, smooth=1):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        bce_loss = self.bce(inputs_flat, targets_flat)

        inputs_sig = torch.sigmoid(inputs_flat)
        intersection = (inputs_sig * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.sum() + targets_flat.sum() + smooth)

        return 0.7 * bce_loss + 0.3 * dice_loss


# U-Net 模型结构
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
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2));
            self.ups.append(DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2)
        self.final = nn.Conv2d(feats[0], out_c, 1)

    def forward(self, x):
        skips = [];
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x);
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x);
            s = skips[i // 2]
            if x.shape != s.shape: x = TF.resize(x, s.shape[2:])
            x = self.ups[i + 1](torch.cat((s, x), 1))
        return self.final(x)


# 数据集
class PomegranateDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        basename = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, basename + ".png")
        if not os.path.exists(mask_path): mask_path = os.path.join(self.mask_dir, basename + ".jpg")

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


# 流程控制
def train_one_epoch(model, loader, opt, loss_fn, ep, total_ep):
    model.train()
    loss_sum = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f"Epoch [{ep + 1}/{total_ep}] Loss: {loss_sum / len(loader):.4f}")


def generate_pseudo(teacher):
    print("\n正在挖掘机械损伤伪标签...")
    if os.path.exists(PSEUDO_ROOT): shutil.rmtree(PSEUDO_ROOT)
    os.makedirs(PSEUDO_IMG_DIR, exist_ok=True)
    os.makedirs(PSEUDO_MASK_DIR, exist_ok=True)

    real_imgs = set(os.listdir(REAL_IMG_DIR))
    cands = [p for p in glob.glob(os.path.join(UNLABELED_IMG_DIR, "*.*"))
             if os.path.basename(p) not in real_imgs and p.lower().endswith(('.jpg', '.png'))]

    teacher.eval()
    cnt = 0
    with torch.no_grad():
        for p in cands:
            if cnt >= MAX_PSEUDO_SAMPLES: break
            img = cv2.imread(p)
            if img is None: continue
            h, w = img.shape[:2]
            x = TF.resize(TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0), [256, 256]).to(DEVICE)

            prob = torch.sigmoid(teacher(x))
            mask = (prob > 0.5).float()

            if mask.sum() > 10:
                conf = prob[mask == 1].mean().item()
                if conf > CONFIDENCE_THRESHOLD:
                    shutil.copy(p, os.path.join(PSEUDO_IMG_DIR, os.path.basename(p)))
                    mask_np = TF.resize(mask, [h, w],
                                        interpolation=TF.InterpolationMode.NEAREST).squeeze().cpu().numpy() * 255
                    cv2.imwrite(os.path.join(PSEUDO_MASK_DIR, os.path.splitext(os.path.basename(p))[0] + ".png"),
                                mask_np.astype(np.uint8))
                    cnt += 1

    print(f"挖掘结束，共生成{cnt}个伪标签。")
    return cnt


def main():
    print(f"Task 4训练启动 (Device: {DEVICE})")

    # Phase 1
    print("=== Phase 1: Teacher Training ===")
    teacher = SimpleUNet().to(DEVICE)
    opt = optim.Adam(teacher.parameters(), lr=LR)
    crit = WeightedLoss()

    ds_real = PomegranateDataset(REAL_IMG_DIR, REAL_MASK_DIR, True)
    dl_real = DataLoader(ds_real, batch_size=BATCH_SIZE, shuffle=True)
    for e in range(TEACHER_EPOCHS):
        train_one_epoch(teacher, dl_real, opt, crit, e, TEACHER_EPOCHS)
    torch.save(teacher.state_dict(), TEACHER_MODEL_PATH)

    # Phase 2
    n = generate_pseudo(teacher)

    # Phase 3
    if n > 0:
        print("=== Phase 2: Student Training ===")
        ds_pseudo = PomegranateDataset(PSEUDO_IMG_DIR, PSEUDO_MASK_DIR, True)
        ds_final = ConcatDataset([ds_real, ds_pseudo])
        dl_final = DataLoader(ds_final, batch_size=BATCH_SIZE, shuffle=True)
        student = SimpleUNet().to(DEVICE)
        student.load_state_dict(torch.load(TEACHER_MODEL_PATH))
        opt_s = optim.Adam(student.parameters(), lr=LR)
        for e in range(STUDENT_EPOCHS):
            train_one_epoch(student, dl_final, opt_s, crit, e, STUDENT_EPOCHS)
        torch.save(student.state_dict(), STUDENT_MODEL_PATH)
    else:
        print("未生成伪标签，复制Teacher模型。")
        shutil.copy(TEACHER_MODEL_PATH, STUDENT_MODEL_PATH)

    print(f"训练完成。最终模型已保存为: {STUDENT_MODEL_PATH}")


if __name__ == "__main__":
    main()