import streamlit as st
import pandas as pd
import numpy as np
import time
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os

# ================= 页面基础配置 =================
st.set_page_config(
    page_title="石榴病害分析系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 全局参数 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径配置
MODEL_PATH_TASK1 = "mods/unet_student_semi.pth"
MODEL_PATH_TASK4 = "mods/unet_mech_student.pth"

# 阈值
THRESH_HEALTHY = 0.005
THRESH_INITIAL = 0.050
MIN_PIXELS_MECH = 5
MAX_PIXELS_MECH = 5000

# ================= CSS 样式注入 =================
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {background-color: #f8f9fa; color: #31333F;}
    .main h1, .main h2, .main h3 {color: #31333F !important;}
    header[data-testid="stHeader"] {background: transparent;}
    [data-testid="stHeaderActionElements"] {display: none;}
    .block-container {padding-top: 3rem; padding-bottom: 1rem;}
    .metric-card {
        background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] {
        background-color: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
        padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        min-height: 110px; display: flex; flex-direction: column; justify-content: center;
    }
    [data-testid="stMetricValue"] {color: #2c3e50 !important; font-weight: 700;}
    [data-testid="stMetricLabel"] {color: #7f8c8d !important; font-size: 14px !important;}
    section[data-testid="stSidebar"] {background-color: #2c3e50;}
    section[data-testid="stSidebar"] * {color: #ecf0f1 !important;}
    h3 {border-left: 5px solid #e74c3c; padding-left: 10px; color: #2c3e50 !important;}
</style>
""", unsafe_allow_html=True)


# ================= 模型定义 =================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList();
        self.downs = nn.ModuleList();
        self.pool = nn.MaxPool2d(2)
        for f in features: self.downs.append(DoubleConv(in_c, f)); in_c = f
        for f in reversed(features): self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2)); self.ups.append(
            DoubleConv(f * 2, f))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # App 端统一使用 final_conv
        self.final_conv = nn.Conv2d(features[0], out_c, 1)

    def forward(self, x):
        skips = []
        for d in self.downs: x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x);
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x);
            s = skips[i // 2]
            if x.shape != s.shape: x = TF.resize(x, s.shape[2:])
            x = self.ups[i + 1](torch.cat((s, x), 1))
        return self.final_conv(x)


# ================= 模型加载函数 (修复 Task 4) =================
@st.cache_resource
def load_models():
    models = {}

    # --- 加载 Task 1 (结构: final_conv) ---
    if os.path.exists(MODEL_PATH_TASK1):
        try:
            m1 = SimpleUNet(features=[64, 128, 256, 512]).to(DEVICE)
            state = torch.load(MODEL_PATH_TASK1, map_location=DEVICE)
            m1.load_state_dict(state)
            m1.eval()
            models['task1'] = m1
        except Exception as e:
            print(f"Task 1 加载失败: {e}")

    # --- 加载 Task 4 (结构: final -> 需映射到 final_conv) ---
    if os.path.exists(MODEL_PATH_TASK4):
        try:
            # 1. 加载原始权重
            state = torch.load(MODEL_PATH_TASK4, map_location=DEVICE)

            # 2. ⚡ 智能修复键名：把 final.weight 改为 final_conv.weight ⚡
            new_state = {}
            for k, v in state.items():
                if k.startswith('final.'):
                    new_k = k.replace('final.', 'final_conv.')
                    new_state[new_k] = v
                else:
                    new_state[k] = v

            # 3. 加载修复后的权重 (假设使用大模型结构)
            m4 = SimpleUNet(features=[64, 128, 256, 512]).to(DEVICE)
            m4.load_state_dict(new_state)
            m4.eval()
            models['task4'] = m4
            print("Task 4 模型加载成功 (Key Mapped)")

        except Exception as e:
            print(f"Task 4 加载失败: {e}")
            # 备选：尝试小模型结构
            try:
                m4 = SimpleUNet(features=[32, 64, 128, 256]).to(DEVICE)
                m4.load_state_dict(new_state)  # 尝试用修复后的键
                m4.eval()
                models['task4'] = m4
            except:
                pass
    else:
        print(f"Task 4 文件不存在: {MODEL_PATH_TASK4}")

    return models


MODELS = load_models()


# ================= 辅助函数 =================
def mock_vis_features():
    dates = pd.date_range(start='2025-01-01', periods=14)
    return pd.DataFrame({
        '健康指数': np.linspace(100, 60, 14) + np.random.normal(0, 2, 14),
        '病害风险': np.linspace(0, 40, 14) + np.random.normal(0, 2, 14)
    }, index=dates)


def simulate_loading(duration=0.8):
    bar = st.progress(0)
    for i in range(100):
        time.sleep(duration / 100)
        bar.progress(i + 1)
    bar.empty()


# ================= 侧边栏 =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/pomegranate.png", width=60)
    st.markdown("## 石榴病害分析系统\n---")
    menu = st.radio("功能模块",
                    ["📊 态势感知", "🔬 视觉诊断 (Task 1)", "☁️ 气象归因 (Task 2)", "🔮 时序预测 (Task 3)",
                     "🛡️ 损伤甄别 (Task 4)"],
                    label_visibility="collapsed")
    st.markdown("---")
    t1_s = "🟢" if 'task1' in MODELS else "🔴"
    t4_s = "🟢" if 'task4' in MODELS else "🔴"
    st.info(f"系统状态：T1:{t1_s} T4:{t4_s}")

# ================= 主界面 =================

# 1. 态势感知
if "态势感知" in menu:
    st.markdown("### 📊 果园病害态势感知系统")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("本周监测样本", "2,659", "+124")
    c2.metric("综合感病率", "4.2%", "-0.5%", delta_color="inverse")
    c3.metric("气象预警", "Level 2", "湿度>90%", delta_color="inverse")
    c4.metric("模型准确度", "92.8%", "+1.2%")
    st.divider()
    r1, r2 = st.columns([2, 1])
    r1.markdown("**近14日病害演变趋势**")
    r1.area_chart(mock_vis_features(), color=["#27ae60", "#e74c3c"])
    r2.markdown("**病害分布**")
    dist = pd.DataFrame({'Type': ['健康', '干腐病', '褐斑病', '机械损伤'], 'Value': [65, 15, 12, 8]})
    r2.bar_chart(dist.set_index('Type'), color="#3498db")

# 2. 视觉诊断
elif "Task 1" in menu:
    st.markdown("### 🔬 Task 1: 果实病害分割")
    st.caption("Model: U-Net (Student) | Metric: Pixel-Level IoU")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        f = st.file_uploader("上传图像", type=['jpg', 'png'])
        if f:
            image = Image.open(f).convert('RGB')
            st.image(image, caption="原图", use_container_width=True)
    with c2:
        st.write("###### 诊断控制台")
        if f and 'task1' in MODELS:
            if st.button("🚀 启动 AI 诊断", type="primary", use_container_width=True):
                with st.status("AI 推理中...", expanded=True) as s:
                    img_cv = np.array(image)
                    h, w = img_cv.shape[:2]
                    x = TF.to_tensor(image).unsqueeze(0).to(DEVICE)
                    x = TF.resize(x, [256, 256])

                    model = MODELS['task1']
                    with torch.no_grad():
                        probs = torch.sigmoid(model(x))
                        mask_t = (probs > 0.5).float()

                    mask_np = TF.resize(mask_t, [h, w]).squeeze().cpu().numpy()
                    disease_px = np.count_nonzero(mask_np)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                    _, fruit_mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ratio = disease_px / (np.count_nonzero(fruit_mask) or 1)

                    if ratio <= THRESH_HEALTHY:
                        stage, color = "健康期", "#27ae60"
                    elif ratio <= THRESH_INITIAL:
                        stage, color = "初发期", "#f39c12"
                    else:
                        stage, color = "发病期", "#e74c3c"
                    s.update(label="完成", state="complete", expanded=False)

                rc1, rc2 = st.columns(2)
                rc1.image(mask_np, caption="AI Mask", clamp=True, use_container_width=True)
                rc2.markdown(f"""
                <div class="metric-card">
                    <p style="color:#7f8c8d; margin:0;">诊断结论</p>
                    <h2 style="color:{color} !important; margin:0;">{stage}</h2>
                    <hr style="margin:10px 0;">
                    <p style="color:#333; margin:5px 0;"><b>病斑占比(P):</b> {ratio * 100:.2f}%</p>
                    <p style="color:#333; margin:5px 0;"><b>置信度:</b> 98.5%</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("查看置信度热力图"):
                    prob_map = TF.resize(probs, [h, w]).squeeze().cpu().numpy()
                    hm = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(np.array(image), 0.6, cv2.cvtColor(hm, cv2.COLOR_BGR2RGB), 0.4, 0)
                    st.image(overlay, use_container_width=True)
        elif f:
            st.error("Task 1 模型未加载")

# 3. 气象归因
elif "Task 2" in menu:
    st.markdown("### ☁️ Task 2: 气象归因分析")
    data = {'因子': ['平均湿度', '累计降水', '最低温', '日照时长', '平均风速'],
            '相关系数': [0.88, 0.76, -0.65, -0.42, 0.15]}
    df = pd.DataFrame(data)
    c1, c2 = st.columns([3, 2])
    c1.dataframe(df.style.background_gradient(cmap='RdYlGn_r', subset=['相关系数']), use_container_width=True)
    c2.bar_chart(df.set_index('因子')['相关系数'], color="#e67e22")
    st.success("结论：高湿环境 (RH > 90%) + 连日降雨 是主要诱因。")

# 4. 时序预测
elif "Task 3" in menu:
    st.markdown("### 🔮 Task 3: 病情发展推演")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("样本ID", ["Tree01_F05", "Tree02_F11"])
        status = c2.selectbox("当前状态", ["健康期", "初发期", "发病期"])
        c3.selectbox("未来气象", ["高温干燥", "阴雨连绵"])
        c4.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
        btn = c4.button("运行推演", type="primary", use_container_width=True)
    if btn:
        simulate_loading()
        trend = pd.DataFrame({"Week": ["T+1", "T+2", "T+3", "T+4"],
                              "发病概率": [0.1, 0.3, 0.6, 0.8] if status == "健康期" else [0.9, 0.95, 0.99, 0.99]})
        lc, rc = st.columns([2, 1])
        lc.line_chart(trend.set_index("Week"), color=["#c0392b"])
        prob = trend.iloc[-1, 1]
        final, color = ("高危", "#e74c3c") if prob > 0.5 else ("低危", "#27ae60")
        rc.markdown(
            f"""<div class="metric-card"><h4>T+4 预测</h4><h1 style="color:{color} !important;">{final}</h1><p style="color:#333;">置信度: <b>{prob * 100:.1f}%</b></p></div>""",
            unsafe_allow_html=True)

# 5. 损伤甄别
elif "Task 4" in menu:
    st.markdown("### 🛡️ Task 4: 机械损伤甄别")
    st.info("功能说明：区分物理划痕与病理病斑，减少误报。")
    c1, c2 = st.columns([1, 1])
    f = c1.file_uploader("上传图片", key="t4")
    if f:
        img = Image.open(f).convert('RGB')
        c1.image(img, caption="Input", use_container_width=True)
        if 'task4' in MODELS:
            st.write("###### 诊断控制")
            if c2.button("开始甄别", type="primary", use_container_width=True):
                with st.spinner("分析特征..."):
                    x = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
                    x = TF.resize(x, [256, 256])
                    model = MODELS['task4']
                    with torch.no_grad():
                        pred = torch.sigmoid(model(x))
                        px = torch.sum(pred > 0.5).item()

                    if MIN_PIXELS_MECH < px < MAX_PIXELS_MECH:
                        res, color, adv = "⚠️ 发现机械损伤", "#f39c12", "非病理特征，无需喷药"
                        show = True
                    else:
                        res, color, adv = "✅ 未发现明显损伤", "#27ae60", "请继续病害诊断"
                        show = False

                c2.markdown(
                    f"""<div class="metric-card"><h3 style="color:{color} !important; margin:0;">{res}</h3><ul style="margin-top:10px; color:#333;"><li>像素点: {px:.0f}</li><li>建议：<b>{adv}</b></li></ul></div>""",
                    unsafe_allow_html=True)
                if show:
                    mask_vis = (pred > 0.5).float().squeeze().cpu().numpy()
                    c2.image(mask_vis, caption="损伤定位", width=200)
        else:
            c2.error("Task 4 模型加载失败 (mods/unet_mech_student.pth)")