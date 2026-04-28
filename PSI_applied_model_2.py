import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import os

warnings.filterwarnings('ignore')


# ==========================================
# 1. PSI 计算核心逻辑 (保持不变)
# ==========================================
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


# ==========================================
# 2. 数据与外部基线加载
# ==========================================
print("正在加载主数据集与静态基线结果...")
df = pd.read_parquet('dataset/UGR16_drift_dataset.parquet')
static_csv_path = 'dataset/static_baseline_results.csv'

if not os.path.exists(static_csv_path):
    raise FileNotFoundError(f"未找到静态基线结果，请先运行静态代码导出！")

static_df = pd.read_csv(static_csv_path)
static_df['time_window'] = pd.to_datetime(static_df['time_window'])

# ==========================================
# 3. 初始化模型与【锚点记忆池】
# ==========================================
start_time = pd.to_datetime('2016-07-27 13:43:00')
split_time = pd.to_datetime('2016-07-27 19:43:00')
end_time = df['time_window'].max()
drop_cols = ['Source IP', 'time_window', 'target_label']

train_df = df[(df['time_window'] >= start_time) & (df['time_window'] < split_time)]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target_label']

print("训练初始自适应模型...")
# 核心改动 1：初始树数量从 10000 降为 1000，既能充分拟合基线，又能防止过拟合，且极大提升运行速度
model_adaptive = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model_adaptive.fit(X_train, y_train)

# 记忆池：依然提供最多5000条攻击+5000条正常，作为稳定基石
attacks_memory = train_df[train_df['target_label'] == 1].copy()
if len(attacks_memory) > 5000: attacks_memory = attacks_memory.sample(n=5000, random_state=42)
normals_memory = train_df[train_df['target_label'] == 0].copy()
if len(normals_memory) > 5000: normals_memory = normals_memory.sample(n=5000, random_state=42)

memory_df = pd.concat([attacks_memory, normals_memory], axis=0)
X_memory = memory_df.drop(columns=drop_cols)
y_memory = memory_df['target_label']

# 提取关键特征用于监控
top_10_features = pd.Series(model_adaptive.feature_importances_, index=X_train.columns).nlargest(10).index.tolist()
base_distribution_data = train_df[top_10_features].copy()

# ==========================================
# 4. 全量数据 + 增量流式评估流水线 (流式学习核心)
# ==========================================
eval_step = pd.Timedelta('20min')
psi_monitor_window = pd.Timedelta('1h')
sliding_retrain_window = pd.Timedelta('12h')

current_time = split_time
monitor_start_time = split_time

time_stamps_adaptive = []
f1_scores_adaptive = []
retrain_events = []

print("\n--- 开始终极自适应流式评估 (增量学习微调版) ---")
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
                        print(
                            f"[{next_time.strftime('%m-%d %H:%M')}] PSI 报警，但近期攻击仅 {len(recent_attacks)} 条。拒绝重训！")
                    else:
                        print(
                            f"\n🚀 [{next_time.strftime('%m-%d %H:%M')}] PSI: {current_psi:.4f} | 启动增量自适应微调...")

                        X_recent = retrain_df.drop(columns=drop_cols)
                        y_recent = retrain_df['target_label']

                        # 融合近期全量数据与记忆池
                        X_retrain_mixed = pd.concat([X_recent, X_memory], axis=0)
                        y_retrain_mixed = pd.concat([y_recent, y_memory], axis=0)

                        # 计算合理的动态权重
                        num_pos = y_retrain_mixed.sum()
                        num_neg = len(y_retrain_mixed) - num_pos
                        calculated_weight = num_neg / num_pos if num_pos > 0 else 1.0
                        capped_weight = min(calculated_weight, 30.0)

                        print(f"  -> 混合总样本: {len(X_retrain_mixed)} 条 | 实际施加权重: {capped_weight:.1f}")

                        # ========================================================
                        # 核心改动 2：真正的增量学习 (Incremental Learning)
                        # ========================================================
                        # ⚠️ 关键修复：在覆盖变量之前，先提取并保存旧模型的 booster！
                        old_booster = model_adaptive.booster_

                        # 我们不再推翻重建，而是在原有模型的基础上额外增加 100 棵树去学习新的攻击模式
                        new_n_estimators = model_adaptive.n_estimators + 100

                        # 实例化新包装器，指定新的树总数
                        model_adaptive = lgb.LGBMClassifier(
                            n_estimators=new_n_estimators,
                            learning_rate=0.01,  # 稍大的学习率让模型快速关注新样本
                            scale_pos_weight=capped_weight,
                            random_state=42,
                            n_jobs=-1,
                            verbose=-1
                        )

                        # 重点：传入保存好的 old_booster，继承之前所有的树和知识！
                        model_adaptive.fit(
                            X_retrain_mixed,
                            y_retrain_mixed,
                            init_model=old_booster
                        )

                        retrain_events.append(next_time)
                        # 更新基线分布数据，将其对齐到适应后的新分布
                        base_distribution_data = retrain_df[top_10_features].copy()

        monitor_start_time = next_time
    current_time = next_time

# ==========================================
# 5. 绘制世纪对决图
# ==========================================
print("\n正在生成终极对比图表...")
N = 15
sma_static = static_df['f1_score_static'].rolling(window=N, min_periods=1).mean()
sma_adaptive = pd.Series(f1_scores_adaptive).rolling(window=N, min_periods=1).mean()

plt.figure(figsize=(16, 8))
plt.plot(static_df['time_window'], static_df['f1_score_static'], marker='x', markersize=3, color='#d62728', alpha=0.15,
         label='Static Raw F1')
plt.plot(static_df['time_window'], sma_static, color='#d62728', linewidth=2, linestyle='--',
         label='Static Moving Average')
plt.plot(time_stamps_adaptive, f1_scores_adaptive, marker='o', markersize=3, color='#2ca02c', alpha=0.25,
         label='Adaptive Raw F1')
plt.plot(time_stamps_adaptive, sma_adaptive, color='#006400', linewidth=3, label='Adaptive Moving Average')

for i, rt_time in enumerate(retrain_events):
    label = 'Incremental Drift Update' if i == 0 else ""
    plt.axvline(x=rt_time, color='#9467bd', linestyle='-.', linewidth=1.5, alpha=0.9, label=label)

plt.axvline(x=split_time, color='black', linestyle='-', linewidth=2, label='Train/Test Split')
plt.axvspan(start_time, split_time, color='grey', alpha=0.1)

plt.title("Head-to-Head: Incremental Adaptive Retraining vs. Static Baseline", fontsize=18, fontweight='bold', pad=15)
plt.xlabel('Time (MM-DD HH:MM)', fontsize=13)
plt.ylabel('F1 Score', fontsize=13)
plt.ylim(-0.05, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=11, ncol=2)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()