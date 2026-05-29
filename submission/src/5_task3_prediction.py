import os
import re
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

# --- 配置区域 ---
IMAGE_DIR = r"datasets/all_images"       # 图片文件夹路径
RESULT1_FILE = "result1.xlsx"            # 任务1生成的标签文件
WEATHER_FEATS = "weather_features.xlsx"  # 任务2生成的气象特征
ATTACHMENT_3 = "附件3"                    # 待预测文件夹
OUTPUT_FILE = "result3.xlsx"             # 结果输出路径

PSEUDO_LABEL = True  # 是否开启半监督伪标签
CONF_THRESH = 0.90  # 伪标签置信度阈值


# --- 视觉特征提取器 (ResNet18) ---
class VisualExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.model = models.resnet18(pretrained=True)

        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device).eval()

        self.trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, path):
        try:
            t = self.trans(Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # 输出维度: 512
                return self.model(t).flatten().cpu().numpy()
        except:
            return np.zeros(512)


# --- 集成模型定义 (Stacking) ---
def get_model():
    # 使用GBDT+随机森林+SVM进行软投票集成
    return VotingClassifier(estimators=[
        ('gbdt', GradientBoostingClassifier(n_estimators=100, max_depth=3)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC(probability=True))
    ], voting='soft', weights=[2, 1, 1])


def parse_week(fname):
    # 从文件名解析周数
    m = re.search(r'week_(\d+)|w(\d+)', fname, re.IGNORECASE)
    return int(m.group(1) or m.group(2)) if m else 0


def main():
    # 1. 加载数据
    df_lbl = pd.read_excel(RESULT1_FILE)
    df_wea = pd.read_excel(WEATHER_FEATS)

    # 标签映射: 健康=0, 初发=1, 发病=2
    df_lbl['Label'] = df_lbl['果实阶段'].map({'健康期': 0, '初发期': 1, '发病期': 2})
    df_lbl['Week'] = df_lbl['周数'].apply(parse_week)

    # 2. 构建气象特征字典 (周数 -> 特征向量)
    w_cols = [c for c in df_wea.columns if c.endswith('_7d')]
    w_week_col = next((c for c in df_wea.columns if '周' in c or 'week' in c.lower()), None)
    df_wea['Wk'] = df_wea[w_week_col].apply(lambda x: parse_week(str(x))) if w_week_col else 0
    wea_map = df_wea.groupby('Wk')[w_cols].mean().to_dict('index')

    ext = VisualExtractor()
    feat_cache = {}
    X, y = [], []

    # 3. 构建训练集
    # 遍历所有已标注数据
    for _, row in df_lbl.dropna(subset=['Label']).iterrows():
        t, f = str(row['果实编号']).split('.')
        fname = f"w{row['Week']:02d}_t{int(t):02d}_f{int(f):02d}.jpg"
        path = os.path.join(IMAGE_DIR, fname)

        if not os.path.exists(path):
            path = path.replace('.jpg', '.png')
        if not os.path.exists(path):
            continue

        # 提取或读取缓存的特征
        if fname not in feat_cache:
            feat_cache[fname] = ext.extract(path)

        vis = feat_cache[fname]
        wea = list(wea_map.get(row['Week'], {c: 0 for c in w_cols}).values())
        base = np.concatenate([vis, wea])

        # 样本构造:当前时刻(Delta=0)
        X.append(np.append(base, 0))
        y.append(row['Label'])

    #     if row['Label'] == 2:
    #         for d in [1, 2, 3, 4]:
    #             X.append(np.append(base, d))
    #             y.append(2)
    #
    # X, y = np.array(X), np.array(y)

    # 4. 训练初始模型
    model = get_model()
    model.fit(X, y)

    # 5. 伪标签挖掘 (半监督学习)
    if PSEUDO_LABEL:
        p_X, p_y = [], []
        # 遍历所有未标注图片
        for p in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
            if os.path.basename(p) in feat_cache: continue

            wk = parse_week(os.path.basename(p))
            if wk == 0: continue

            # 提取特征并预测
            base = np.concatenate([ext.extract(p), list(wea_map.get(wk, {c: 0 for c in w_cols}).values())])
            prob = model.predict_proba(np.append(base, 0).reshape(1, -1))[0]

            # 高置信度筛选
            if max(prob) > CONF_THRESH:
                lbl = np.argmax(prob)
                p_X.append(np.append(base, 0))
                p_y.append(lbl)
                # 对伪标签应用逻辑增强
                if lbl == 2:
                    for d in [1, 2, 3, 4]:
                        p_X.append(np.append(base, d))
                        p_y.append(2)

        # 合并数据并重训
        if p_X:
            model.fit(np.concatenate([X, p_X]), np.concatenate([y, p_y]))

    # 6. 预测附件3 (未来推演)
    res = []
    # 递归搜索附件3中的所有图片
    for p in glob.glob(os.path.join(ATTACHMENT_3, "**", "*.*"), recursive=True):
        if not p.lower().endswith(('.jpg', '.png')): continue

        wk = parse_week(os.path.basename(p)) or 10
        base = np.concatenate([ext.extract(p), list(wea_map.get(wk, {c: 0 for c in w_cols}).values())])

        # 解析果实编号
        fid = os.path.basename(p)
        try:
            parts = os.path.basename(p).split('_')[-1].split('-')
            fid = f"{parts[0]}.{parts[1].split('.')[0]}"
        except:
            pass

        # 预测未来1-4周
        for d in [1, 2, 3, 4]:
            pred = model.predict(np.append(base, d).reshape(1, -1))[0]
            res.append({
                "周数": f"week_{wk + d:02d}",
                "果实编号": fid,
                "病害阶段": {0: '健康期', 1: '初发期', 2: '发病期'}[pred]
            })

    # 保存结果
    pd.DataFrame(res).to_excel(OUTPUT_FILE, index=False)
    print(f"任务三完成，结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()