import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. 读取与评估逻辑 (保持原样) ---
df = pd.read_parquet('dataset/UGR16_drift_dataset.parquet')
start_time = pd.to_datetime('2016-07-27 13:43:00')
split_time = pd.to_datetime('2016-07-28 13:43:00')
end_time = df['time_window'].max()

train_df = df[(df['time_window'] >= start_time) & (df['time_window'] < split_time)]
drop_cols = ['Source IP', 'time_window', 'target_label']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target_label']

model_static = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
model_static.fit(X_train, y_train)

eval_step = pd.Timedelta('20min')
current_time = start_time
time_stamps, f1_scores = [], []

while current_time < end_time:
    next_time = current_time + eval_step
    batch = df[(df['time_window'] >= current_time) & (df['time_window'] < next_time)]
    if len(batch) > 0:
        X_batch = batch.drop(columns=drop_cols)
        y_true = batch['target_label']
        y_pred = model_static.predict(X_batch)
        if y_true.sum() > 0:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            time_stamps.append(current_time)
            f1_scores.append(current_f1)
    current_time = next_time

# --- 2. 新增：从原始 CSV 提取攻击分布 ---
# 请确保路径正确，且原始 CSV 包含 'Date time' 和 'Label' 列
print("正在读取原始攻击分布数据...")
raw_df = pd.read_csv("dataset/UGR_sample_5M.csv", encoding_errors='ignore')
raw_df.columns = raw_df.columns.str.strip()
raw_df['Date time'] = pd.to_datetime(raw_df['Date time'])
raw_df['Label'] = raw_df['Label'].astype(str).str.lower()

# 过滤恶意流量并按 20min 聚合
attacks_only = raw_df[raw_df['Label'] != 'background'].copy()
attacks_only['time_window'] = attacks_only['Date time'].dt.floor('20min')
attack_counts = attacks_only.groupby(['time_window', 'Label']).size().unstack(fill_value=0)

# --- 3. 联合可视化 (上下子图对齐) ---
# 创建 2 行 1 列的画布，共享 X 轴
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.2]})

# A. 绘制上图：F1 分数
f1_series = pd.Series(f1_scores)
sma_scores = f1_series.rolling(window=15, min_periods=1).mean()

ax1.plot(time_stamps, f1_scores, marker='o', markersize=3, color='#1f77b4', alpha=0.3, label='Raw F1')
ax1.plot(time_stamps, sma_scores, color='#ff7f0e', linewidth=2.5, label='F1 Moving Average')
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_ylim(-0.05, 1.1)
ax1.legend(loc='lower left')
ax1.set_title("Static Baseline Analysis: F1 Performance vs. Attack Distribution", fontsize=16, fontweight='bold')

# B. 绘制下图：堆叠攻击直方图
# 使用 ax.bar 绘制，确保时间刻度与 ax1 完美对齐
bottom = np.zeros(len(attack_counts))
for col in attack_counts.columns:
    ax2.bar(attack_counts.index, attack_counts[col], bottom=bottom, label=col, width=0.012) # width约为20min
    bottom += attack_counts[col].values

ax2.set_ylabel('Attack Flow Count', fontsize=12)
ax2.set_xlabel('Time (MM-DD HH:MM)', fontsize=12)
ax2.legend(title="Attack Type", bbox_to_anchor=(1.01, 1), loc='upper left')

# C. 共有修饰：训练/测试分割线与阴影
for ax in [ax1, ax2]:
    ax.axvline(x=split_time, color='#d62728', linestyle='--', linewidth=2, label='Split Line')
    ax.axvspan(start_time, split_time, color='lightgreen', alpha=0.1)
    ax.axvspan(split_time, end_time, color='lightcoral', alpha=0.05)
    ax.grid(True, linestyle=':', alpha=0.5)

# X 轴时间格式优化
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()