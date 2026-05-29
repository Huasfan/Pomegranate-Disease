import os
import pandas as pd
import numpy as np
import glob
import re

# --- 配置区域 ---
RESULT1_FILE = "result1.xlsx"
WEATHER_INPUT_DIR = "附件2"
OUTPUT_CORRELATION_FILE = "result2.xlsx"
OUTPUT_WEATHER_FEATS = "weather_features.xlsx"
WINDOW_DAYS = 7

TARGET_VARS = ['temp', 'dewpt', 'rh', 'precip_rate', 'solar_rad', 'ghi', 'dhi', 'dni', 'pres', 'wind_spd']
IMG_COL_CANDIDATES = ['image', '图像文件名', '____', '_____', '原始文件名']


def clean_col(col):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(col).strip()).lower()


def load_weather():
    all_data = []
    for f in glob.glob(os.path.join(WEATHER_INPUT_DIR, "*.xlsx")):
        try:
            df = pd.read_excel(f)
            filename = os.path.basename(f)
            week_match = re.search(r'(week)_(\d+)', filename, re.IGNORECASE)

            if week_match:
                week_num_str = week_match.group(2).zfill(2)
                df['__wk_source__'] = f"week_{week_num_str}"
            else:
                df['__wk_source__'] = "unknown"

            df.columns = [clean_col(c) for c in df.columns]
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    merged = pd.concat(all_data, ignore_index=True)
    dt_str = merged['date'].astype(str) + ' ' + merged['time'].astype(str)
    merged['datetime'] = pd.to_datetime(dt_str.str.replace('-', ':'), format='%Y:%m:%d %H:%M:%S', errors='coerce')

    img_col = next((c for c in merged.columns if c in IMG_COL_CANDIDATES), None)
    if img_col: merged.rename(columns={img_col: '图像文件名'}, inplace=True)
    return merged


def engineer_features(df):
    df.rename(columns={'__wk_source__': 'wk_key'}, inplace=True)
    df.dropna(subset=['datetime'], inplace=True)
    df.sort_values('datetime', inplace=True)
    df.set_index('datetime', inplace=True)

    feats = df.copy()
    for var in TARGET_VARS:
        if var not in df.columns: continue
        roll = df[var].rolling(f'{WINDOW_DAYS}D', closed='left')
        feats[f'{var}_avg_7d'] = roll.mean()
        feats[f'{var}_max_7d'] = roll.max()
        if var == 'precip_rate': feats[f'{var}_sum_7d'] = roll.sum()
        if var == 'rh': feats['rh_high_count_7d'] = (df[var] > 90).rolling(f'{WINDOW_DAYS}D', closed='left').sum()

    feats.dropna(subset=['图像文件名'], inplace=True)
    feats.reset_index(inplace=True)
    feats['wk_key'] = feats['wk_key'].astype(str)
    feats.to_excel(OUTPUT_WEATHER_FEATS, index=False)
    return feats


def extract_fruit_id(filename):
    if pd.isna(filename): return None
    name = re.sub(r'\.\w+$', '', str(filename))
    parts = name.split('_')
    if len(parts) >= 3 and parts[0].startswith('w'):
        return f"{parts[1][1:].lstrip('0')}.{parts[2][1:].lstrip('0')}"

    match = re.match(r'(\d+)_(\d+)$', name)
    if match: return f"{match.group(1).lstrip('0')}.{match.group(2).lstrip('0')}"
    return None


def main():
    # 1. 处理气象数据
    df_wea = engineer_features(load_weather())

    # 2. 处理 Task1 结果
    df_res = pd.read_excel(RESULT1_FILE)
    df_res['果实编号'] = df_res['果实编号'].astype(str)
    df_res['周数'] = df_res['周数'].astype(str)
    df_res.dropna(subset=['果实阶段'], inplace=True)
    df_res['Stage_Code'] = df_res['果实阶段'].map({'健康期': 0, '初发期': 1, '发病期': 2})

    # 3. 关联匹配
    df_wea['fid_key'] = df_wea['图像文件名'].apply(extract_fruit_id)
    merged = df_wea.merge(df_res, left_on=['wk_key', 'fid_key'], right_on=['周数', '果实编号'], how='inner')
    merged.dropna(subset=TARGET_VARS, inplace=True)

    # 4. 计算相关性
    if len(merged) > 5:
        cols = [c for c in merged.columns if c.endswith('_7d')]
        corr = merged[cols + ['Stage_Code']].corr(method='spearman')[['Stage_Code']].sort_values('Stage_Code',
                                                                                                 ascending=False)
        corr.to_excel(OUTPUT_CORRELATION_FILE)
        print(f"Task 2 已完成. 有效样本数: {len(merged)}， 保存至 {OUTPUT_CORRELATION_FILE}")
    else:
        print(f"Error: 样本数量不足 ({len(merged)})，检查数据")


if __name__ == "__main__":
    main()