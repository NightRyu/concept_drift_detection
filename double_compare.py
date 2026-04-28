import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import os


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


def calculate_recall_at_precision(y_true, y_prob, target_precision=0.90):
    if sum(y_true) == 0: return 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_indices = np.where(precisions >= target_precision)[0]
    if len(valid_indices) == 0: return 0.0
    return max(recalls[valid_indices])


df = pd.read_parquet('dataset/UGR16_drift_dataset.parquet')

start_time = pd.to_datetime('2016-07-27 13:43:00')
split_time = pd.to_datetime('2016-07-28 01:43:00')
end_time = df['time_window'].max()
drop_cols = ['Source IP', 'time_window', 'target_label']

train_df = df[(df['time_window'] >= start_time) & (df['time_window'] < split_time)]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target_label']

print("Static Baseline")
model_static = lgb.LGBMClassifier(n_estimators=10000,
                                  learning_rate=0.01,
                                  random_state=42,
                                  n_jobs=-1,
                                  verbose=-1,
                                  class_weight='balanced'
                                  )
model_static.fit(X_train, y_train)

print("Initial Adaptive")
model_adaptive = copy.deepcopy(model_static)
model_adaptive.fit(X_train, y_train)

attacks_memory = train_df[train_df['target_label'] == 1].copy()
if len(attacks_memory) > 5000: attacks_memory = attacks_memory.sample(n=5000, random_state=42)
normals_memory = train_df[train_df['target_label'] == 0].copy()
if len(normals_memory) > 5000: normals_memory = normals_memory.sample(n=5000, random_state=42)

memory_df = pd.concat([attacks_memory, normals_memory], axis=0)
X_memory = memory_df.drop(columns=drop_cols)
y_memory = memory_df['target_label']

top_10_features = pd.Series(model_adaptive.feature_importances_, index=X_train.columns).nlargest(10).index.tolist()
base_distribution_data = train_df[top_10_features].copy()

eval_step = pd.Timedelta('20min')
psi_monitor_window = pd.Timedelta('1h')
sliding_retrain_window = pd.Timedelta('6h')

current_time = start_time
monitor_start_time = split_time

time_stamps = []
f1_scores_static = []
f1_scores_adaptive = []
retrain_events = []
all_predictions = []

while current_time < end_time:
    next_time = current_time + eval_step
    batch = df[(df['time_window'] >= current_time) & (df['time_window'] < next_time)]

    if len(batch) > 0:
        X_batch = batch.drop(columns=drop_cols)
        y_true = batch['target_label']

        y_pred_static = model_static.predict(X_batch)
        y_pred_adaptive = model_adaptive.predict(X_batch)

        y_prob_static = model_static.predict_proba(X_batch)[:, 1]
        y_prob_adaptive = model_adaptive.predict_proba(X_batch)[:, 1]

        if y_true.sum() > 0:
            time_stamps.append(current_time)
            f1_scores_static.append(f1_score(y_true, y_pred_static, zero_division=0))
            f1_scores_adaptive.append(f1_score(y_true, y_pred_adaptive, zero_division=0))

        batch_results = pd.DataFrame({
            'time_window': current_time,
            'y_true': y_true.values,
            'y_prob_static': y_prob_static,
            'y_prob_adaptive': y_prob_adaptive
        })
        all_predictions.append(batch_results)

    # PSI monitor and retrain for dynamic model
    if next_time - monitor_start_time >= psi_monitor_window:
        curr_monitor_data = df[(df['time_window'] >= monitor_start_time) & (df['time_window'] < next_time)]

        if len(curr_monitor_data) > 20:
            is_drift, current_psi = check_concept_drift(
                base_distribution_data,
                curr_monitor_data,
                top_10_features,
                threshold=0.004
            )

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
                            learning_rate=0.01,
                            random_state=42,
                            n_jobs=-1,
                            verbose=-1
                        )
                        model_adaptive.fit(X_retrain_mixed, y_retrain_mixed)

                        retrain_events.append(next_time)
                        base_distribution_data = retrain_df[top_10_features].copy()

        monitor_start_time = next_time
    current_time = next_time

preds_df = pd.concat(all_predictions, ignore_index=True)

# Global metrics calculate
test_preds = preds_df[preds_df['time_window'] >= start_time]
if test_preds['y_true'].sum() > 0:
    g_pr_static = average_precision_score(test_preds['y_true'], test_preds['y_prob_static'])
    g_pr_adapt = average_precision_score(test_preds['y_true'], test_preds['y_prob_adaptive'])
    g_rec_static = calculate_recall_at_precision(test_preds['y_true'], test_preds['y_prob_static'], 0.90)
    g_rec_adapt = calculate_recall_at_precision(test_preds['y_true'], test_preds['y_prob_adaptive'], 0.90)

    print(
        f"Global PR-AUC          | Static: {g_pr_static:.4f}  ->  Adaptive: {g_pr_adapt:.4f}  [increase: {(g_pr_adapt - g_pr_static) / g_pr_static * 100:.1f}%]")
    print(
        f"Global Recall@90% Prec | Static: {g_rec_static:.4f}  ->  Adaptive: {g_rec_adapt:.4f}  [increase: {(g_rec_adapt - g_rec_static) / (g_rec_static + 1e-5) * 100:.1f}%]")
eval_freq = pd.Timedelta('1h')
rolling_eval_window = pd.Timedelta('6h')

metric_times = []
pr_aucs_adaptive, pr_aucs_static = [], []
recalls_adaptive, recalls_static = [], []

metric_time = start_time + rolling_eval_window
while metric_time <= end_time:
    window_start = metric_time - rolling_eval_window
    window_data = preds_df[(preds_df['time_window'] >= window_start) & (preds_df['time_window'] < metric_time)]

    if window_data['y_true'].sum() > 20:
        y_t = window_data['y_true']
        # PR-AUC
        pr_aucs_static.append(average_precision_score(y_t, window_data['y_prob_static']))
        pr_aucs_adaptive.append(average_precision_score(y_t, window_data['y_prob_adaptive']))

        # Recall @ 90% Precision
        recalls_static.append(calculate_recall_at_precision(y_t, window_data['y_prob_static'], 0.90))
        recalls_adaptive.append(calculate_recall_at_precision(y_t, window_data['y_prob_adaptive'], 0.90))

        metric_times.append(metric_time)
    metric_time += eval_freq

raw_df = pd.read_csv("dataset/UGR_sample_5M.csv", encoding_errors='ignore')
raw_df.columns = raw_df.columns.str.strip()
raw_df['Date time'] = pd.to_datetime(raw_df['Date time'])
raw_df['Label'] = raw_df['Label'].astype(str).str.lower()
attacks_only = raw_df[raw_df['Label'] != 'background'].copy()
attacks_only['time_window'] = attacks_only['Date time'].dt.floor('20min')
attack_counts = attacks_only.groupby(['time_window', 'Label']).size().unstack(fill_value=0)

N = 15
sma_static = pd.Series(f1_scores_static).rolling(window=N, min_periods=1).mean()
sma_adaptive = pd.Series(f1_scores_adaptive).rolling(window=N, min_periods=1).mean()

fig1, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})

ax1_top.plot(time_stamps, f1_scores_static, marker='x', markersize=3, color='#d62728', alpha=0.15, label='Static Raw F1')
ax1_top.plot(time_stamps, sma_static, color='#d62728', linewidth=2, linestyle='--', label='Static Moving Average')
ax1_top.plot(time_stamps, f1_scores_adaptive, marker='o', markersize=3, color='#2ca02c', alpha=0.25, label='Adaptive Raw F1')
ax1_top.plot(time_stamps, sma_adaptive, color='#006400', linewidth=3, label='Adaptive Moving Average')

ax1_top.set_title("Stream Evaluation: F1 Recovery & Attack Distribution", fontsize=18, fontweight='bold', pad=15)
ax1_top.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1_top.set_ylim(-0.05, 1.1)
ax1_top.legend(loc='upper right', fontsize=11, ncol=2)

bottom = np.zeros(len(attack_counts))
for col in attack_counts.columns:
    ax1_bot.bar(attack_counts.index, attack_counts[col], bottom=bottom, label=col, width=0.012)
    bottom += attack_counts[col].values

ax1_bot.set_ylabel('Attack Flow Count', fontsize=12, fontweight='bold')
ax1_bot.set_xlabel('Time (MM-DD HH:MM)', fontsize=13, fontweight='bold')
ax1_bot.legend(title="Attack Type", bbox_to_anchor=(1.01, 1), loc='upper left')

for ax in [ax1_top, ax1_bot]:
    for rt_time in retrain_events:
        ax.axvline(x=rt_time, color='#9467bd', linestyle='-.', linewidth=1.5, alpha=0.9)
    ax.axvline(x=split_time, color='black', linestyle='-', linewidth=2)
    ax.axvspan(start_time, split_time, color='grey', alpha=0.1)
    ax.grid(True, linestyle=':', alpha=0.6)

ax1_bot.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1_bot.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
fig1.autofmt_xdate(rotation=45)
fig1.tight_layout()

plt.show()

fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

ax2_top.plot(metric_times, pr_aucs_static, color='#d62728', linewidth=2.5, linestyle='--', label='Static PR-AUC')
ax2_top.plot(metric_times, pr_aucs_adaptive, color='#006400', linewidth=3, marker='o', markersize=5, label='Adaptive PR-AUC')

ax2_top.set_title("Advanced Robustness Metrics", fontsize=18, fontweight='bold', pad=15)
ax2_top.set_ylabel('PR-AUC Score', fontsize=12, fontweight='bold')
ax2_top.set_ylim(-0.05, 1.1)
ax2_top.legend(loc='lower left', fontsize=11)

ax2_bot.plot(metric_times, recalls_static, color='#ff7f0e', linewidth=2.5, linestyle='--', label='Static Recall@90% Prec')
ax2_bot.plot(metric_times, recalls_adaptive, color='#1f77b4', linewidth=3, marker='s', markersize=5, label='Adaptive Recall@90% Prec')

ax2_bot.set_ylabel('Recall Score\n(at 90% Precision)', fontsize=12, fontweight='bold')
ax2_bot.set_xlabel('Time (MM-DD HH:MM)', fontsize=13, fontweight='bold')
ax2_bot.set_ylim(-0.05, 1.1)
ax2_bot.legend(loc='lower left', fontsize=11)

for ax in [ax2_top, ax2_bot]:
    for rt_time in retrain_events:
        ax.axvline(x=rt_time, color='#9467bd', linestyle='-.', linewidth=1.5, alpha=0.9)
    ax.axvline(x=split_time, color='black', linestyle='-', linewidth=2)
    ax.axvspan(start_time, split_time, color='grey', alpha=0.1)
    ax.grid(True, linestyle=':', alpha=0.6)

ax2_bot.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2_bot.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
fig2.autofmt_xdate(rotation=45)
fig2.tight_layout()

plt.show()