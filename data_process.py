import pandas as pd
import numpy as np

file_path = "dataset/UGR_sample_5M.csv"
df = pd.read_csv(file_path)

# Data cleaning
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# process timestamp and label
df['Timestamp'] = pd.to_datetime(df['Date time'])
df = df.sort_values('Timestamp').reset_index(drop=True)

df['Label'] = df['Label'].astype(str).str.lower()
df['is_malicious'] = (df['Label'] != 'background').astype(int)

# Extract attacker's IP
malicious_ips = df[df['Label'] != 'background']['Source IP'].unique()
all_ips = df['Source IP'].unique()
benign_ips = np.setdiff1d(all_ips, malicious_ips)

df['Flow_Packets_s'] = df['Packets'] / (df['Duration'] + 1e-6)
df['Average_Packet_Size'] = df['Bytes'] / df['Packets'].replace(0, 1)

df['Flag'] = df['Flag'].astype(str).str.upper()
df['SYN_Flag'] = df['Flag'].str.contains('S').astype(int)
df['ACK_Flag'] = df['Flag'].str.contains('A').astype(int)
df['PSH_Flag'] = df['Flag'].str.contains('P').astype(int)

# IP-window feature aggregation in minutes level (W = 1)
df['time_window'] = df['Timestamp'].dt.floor('1min')

agg_funcs = {
    'Destination Port': 'nunique',
    'Duration': ['mean', 'max'],
    'Packets': 'sum',
    'Bytes': ['sum', 'mean'],
    'Flow_Packets_s': 'mean',
    'Average_Packet_Size': 'mean',
    'SYN_Flag': 'sum',
    'ACK_Flag': 'sum',
    'PSH_Flag': 'sum',
    'is_malicious': 'max'
}

windowed = df.groupby(['Source IP', 'time_window']).agg(agg_funcs).reset_index()

windowed.columns = ['_'.join(col).strip('_') for col in windowed.columns.values]
windowed.rename(columns={'is_malicious_max': 'malicious_in_window'}, inplace=True)

# Generate label for near-future malicious event prediction (H = 5)
future_calc_df = windowed.sort_values(['Source IP', 'time_window'], ascending=[True, False]).set_index('time_window')
future = future_calc_df.groupby('Source IP')['malicious_in_window'].rolling('5min').max().reset_index()
future.rename(columns={'malicious_in_window': 'target_label'}, inplace=True)
windowed = pd.merge(windowed, future, on=['Source IP', 'time_window'], how='left')

# Calculate 10 min moving averages from history (K = 10)
rolling_features = [
    'Flow_Packets_s_mean',
    'Bytes_mean',
    'Average_Packet_Size_mean',
    'SYN_Flag_sum'
]

history_calc_df = windowed.sort_values(['Source IP', 'time_window']).set_index('time_window')

for feat in rolling_features:
    roll_series = history_calc_df.groupby('Source IP')[feat].rolling('10min').mean()
    roll_df = roll_series.reset_index()
    roll_df.rename(columns={feat: f'{feat}_10m_avg'}, inplace=True)
    windowed = pd.merge(windowed, roll_df, on=['Source IP', 'time_window'], how='left')

final_dataset = windowed.drop(columns=['malicious_in_window']).reset_index()
final_dataset.fillna(0, inplace=True)
final_dataset = final_dataset.sort_values('time_window').reset_index(drop=True)

print("Original Label distribution")
print(df['Label'].value_counts())

conclu = final_dataset['target_label']
print("Target Label distribution")
print(conclu.value_counts())

output_file = 'dataset/UGR16_drift_dataset.parquet'
final_dataset.to_parquet(output_file, index=False)

output_file = 'dataset/UGR16_drift_dataset.csv'
final_dataset.to_csv(output_file, index=False)