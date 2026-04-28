import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os



# calculate PSI
def calculate_psi(expected_array, actual_array, buckets=10):
    expected_array = expected_array[~np.isnan(expected_array)]
    actual_array = actual_array[~np.isnan(actual_array)]
    if len(expected_array) == 0 or len(actual_array) == 0: return 0.0
    bins = np.histogram_bin_edges(expected_array, bins=buckets)
    expected_percents = np.histogram(expected_array, bins=bins)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, bins=bins)[0] / len(actual_array)
    epsilon = 0.0001
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))


def check_concept_drift(base_data, curr_data, features_to_monitor, threshold=0.004):
    total_psi = sum(calculate_psi(base_data[feat].values, curr_data[feat].values) for feat in features_to_monitor)
    avg_psi = total_psi / len(features_to_monitor)
    return avg_psi > threshold, avg_psi



df = pd.read_parquet('dataset/UGR16_drift_dataset.parquet')
static_csv_path = 'dataset/static_baseline_results.csv'
static_df = pd.read_csv(static_csv_path)
static_df['time_window'] = pd.to_datetime(static_df['time_window'])

# init static model
start_time = pd.to_datetime('2016-07-27 13:43:00')
split_time = pd.to_datetime('2016-07-27 19:43:00')
end_time = df['time_window'].max()
drop_cols = ['Source IP', 'time_window', 'target_label']

train_df = df[(df['time_window'] >= start_time) & (df['time_window'] < split_time)]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target_label']

print("训练初始自适应模型...")
model_adaptive = lgb.LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model_adaptive.fit(X_train, y_train)

# memory pool
attacks_memory = train_df[train_df['target_label'] == 1].copy()
if len(attacks_memory) > 5000: attacks_memory = attacks_memory.sample(n=5000, random_state=42)
normals_memory = train_df[train_df['target_label'] == 0].copy()
if len(normals_memory) > 5000: normals_memory = normals_memory.sample(n=5000, random_state=42)

memory_df = pd.concat([attacks_memory, normals_memory], axis=0)
X_memory = memory_df.drop(columns=drop_cols)
y_memory = memory_df['target_label']

top_10_features = pd.Series(model_adaptive.feature_importances_, index=X_train.columns).nlargest(10).index.tolist()
base_distribution_data = train_df[top_10_features].copy()

# ==========================================
# 4. 全量数据 + 受限加权的流式评估流水线
# ==========================================
eval_step = pd.Timedelta('20min')
psi_monitor_window = pd.Timedelta('1h')
sliding_retrain_window = pd.Timedelta('12h')

current_time = split_time
monitor_start_time = split_time

time_stamps_adaptive = []
f1_scores_adaptive = []
retrain_events = []

print("\n--- 开始终极自适应流式评估 (全量保留 + 动态加权) ---")
while current_time < end_time:
    next_time = current_time + eval_step
    batch = df[(df['time_window'] >= current_time) & (df['time_window'] < next_time)]

    if len(batch) > 0:
        X_batch = batch.drop(columns=drop_cols)
        y_true = batch['target_label']
        y_pred_adaptive = model_adaptive.predict(X_batch)
        if y_true.sum() > 0:
            time_stamps_adaptive.append(current_time)
            f1_scores_adaptive.append(f1_score(y_true, y_pred_adaptive, zero_division=0))

    if next_time - monitor_start_time >= psi_monitor_window:
        curr_monitor_data = df[(df['time_window'] >= monitor_start_time) & (df['time_window'] < next_time)]
        if len(curr_monitor_data) > 50:
            is_drift, current_psi = check_concept_drift(base_distribution_data, curr_monitor_data, top_10_features,
                                                        threshold=0.004)

            if is_drift:
                retrain_start = next_time - sliding_retrain_window
                retrain_df = df[(df['time_window'] >= retrain_start) & (df['time_window'] < next_time)]

                if len(retrain_df) > 0:
                    recent_attacks = retrain_df[retrain_df['target_label'] == 1]

                    if len(recent_attacks) < 20:
                        print(f"[{next_time.strftime('%m-%d %H:%M')}] PSI worning, but no enough data to retrain")
                    else:
                        print(
                            f"\n[{next_time.strftime('%m-%d %H:%M')}] PSI: {current_psi:.4f} | retrain model")

                        X_recent = retrain_df.drop(columns=drop_cols)
                        y_recent = retrain_df['target_label']

                        X_retrain_mixed = pd.concat([X_recent, X_memory], axis=0)
                        y_retrain_mixed = pd.concat([y_recent, y_memory], axis=0)

                        model_adaptive = lgb.LGBMClassifier(
                            n_estimators=10000,
                            learning_rate=0.05,
                            random_state=42,
                            n_jobs=-1,
                            verbose=-1
                        )
                        model_adaptive.fit(X_retrain_mixed, y_retrain_mixed)

                        retrain_events.append(next_time)
                        base_distribution_data = retrain_df[top_10_features].copy()

        monitor_start_time = next_time
    current_time = next_time


raw_csv_path = "dataset/UGR_sample_5M.csv"

raw_df = pd.read_csv(raw_csv_path, encoding_errors='ignore')
raw_df.columns = raw_df.columns.str.strip()
raw_df['Date time'] = pd.to_datetime(raw_df['Date time'])
raw_df['Label'] = raw_df['Label'].astype(str).str.lower()

attacks_only = raw_df[raw_df['Label'] != 'background'].copy()
attacks_only['time_window'] = attacks_only['Date time'].dt.floor('20min')
attack_counts = attacks_only.groupby(['time_window', 'Label']).size().unstack(fill_value=0)

# MA
N = 15
sma_static = static_df['f1_score_static'].rolling(window=N, min_periods=1).mean()
sma_adaptive = pd.Series(f1_scores_adaptive).rolling(window=N, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1]})

ax1.plot(static_df['time_window'], static_df['f1_score_static'], marker='x', markersize=3, color='#d62728', alpha=0.15, label='Static Raw F1')
ax1.plot(static_df['time_window'], sma_static, color='#d62728', linewidth=2, linestyle='--', label='Static Moving Average')
ax1.plot(time_stamps_adaptive, f1_scores_adaptive, marker='o', markersize=3, color='#2ca02c', alpha=0.25, label='Adaptive Raw F1')
ax1.plot(time_stamps_adaptive, sma_adaptive, color='#006400', linewidth=3, label='Adaptive Moving Average')

for i, rt_time in enumerate(retrain_events):
    label = 'PSI Drift Retrain Triggered' if i == 0 else ""
    ax1.axvline(x=rt_time, color='#9467bd', linestyle='-.', linewidth=1.5, alpha=0.9, label=label)

ax1.axvline(x=split_time, color='black', linestyle='-', linewidth=2, label='Train/Test Split')
ax1.axvspan(start_time, split_time, color='grey', alpha=0.1)

ax1.set_title("Head-to-Head: Cost-Sensitive Adaptive Retraining vs. Static Baseline", fontsize=18, fontweight='bold', pad=15)
ax1.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=11, ncol=2)

bottom = np.zeros(len(attack_counts))
for col in attack_counts.columns:
    # width 0.012 大约在 matplotlib datetime 中对应不到 20 分钟的视觉宽度
    ax2.bar(attack_counts.index, attack_counts[col], bottom=bottom, label=col, width=0.012)
    bottom += attack_counts[col].values

for rt_time in retrain_events:
    ax2.axvline(x=rt_time, color='#9467bd', linestyle='-.', linewidth=1.5, alpha=0.9)
ax2.axvline(x=split_time, color='black', linestyle='-', linewidth=2)
ax2.axvspan(start_time, split_time, color='grey', alpha=0.1)

ax2.set_ylabel('Attack Flow Count', fontsize=13, fontweight='bold')
ax2.set_xlabel('Time (MM-DD HH:MM)', fontsize=13, fontweight='bold')
ax2.legend(title="Attack Type", bbox_to_anchor=(1.01, 1), loc='upper left') # 图例放右侧防止遮挡
ax2.grid(axis='y', linestyle=':', alpha=0.5)

ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()