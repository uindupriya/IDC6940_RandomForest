import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

label_map = {0: 'Non-Stress', 1: 'Stress'}
palette   = {'Non-Stress': '#378ADD', 'Stress': '#D85A30'}

FS = 4

# -------------------------------------------------------
# HELPER: same engineer_features as RF pipeline
# -------------------------------------------------------
def engineer_features(df, fs=4):
    valid_features = [f for f in ['EDA', 'TEMP'] if f in df.columns]
    win_1min = fs * 60
    win_3min = fs * 60 * 3
    groups = []
    for subject, group in df.groupby('subject'):
        g = group.sort_index().copy()
        for col in valid_features:
            for lag in range(1, 5):
                g[f'{col}_lag_{lag}'] = g[col].shift(lag)
            g[f'{col}_roll1min_mean'] = g[col].rolling(window=win_1min, min_periods=1).mean()
            g[f'{col}_roll1min_std']  = g[col].rolling(window=win_1min, min_periods=2).std()
            g[f'{col}_roll3min_mean'] = g[col].rolling(window=win_3min, min_periods=1).mean()
            g[f'{col}_roll3min_std']  = g[col].rolling(window=win_3min, min_periods=2).std()
        groups.append(g.dropna())
    return pd.concat(groups, ignore_index=True)

# -------------------------------------------------------
# 1. CHEST PAIRPLOT — raw signals, pre-normalization
# -------------------------------------------------------
# 1. CHEST PAIRPLOT — raw signals, pre-normalization
chest = pd.read_csv("wesad_chest_clean.csv")
#label_mapping = {1: 0, 2: 1, 3: 0}
chest['label'] = chest['label'].map(label_map)


sample_c = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in chest.groupby('label')
]).reset_index(drop=True)
sample_c['Condition'] = sample_c['label'].map(label_map)

g = sns.pairplot(
    sample_c[['ECG', 'EDA', 'TEMP', 'RESP', 'Condition']],
    hue='Condition',
    palette=palette,
    plot_kws={'alpha': 0.3, 's': 8},
    diag_kind='kde',
    corner=True
)
g.figure.suptitle('WESAD Chest Signals — Pre-normalization', y=1.01, fontsize=13)
plt.savefig("wesad_chest_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wesad_chest_pairplot.png")

# -------------------------------------------------------
# 2. WRIST — raw EDA vs TEMP (from train split)
# -------------------------------------------------------
# 2. WRIST — raw EDA vs TEMP (from train split)
# -------------------------------------------------------
# 2. WRIST — raw EDA vs TEMP (from train split)
# -------------------------------------------------------

train_df = pd.read_csv("wesad_wrist_train.csv")

# Labels are already binary:
# 0 = Non-Stress
# 1 = Stress

sample_w = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in train_df.groupby('label')
]).reset_index(drop=True)

sample_w['Condition'] = sample_w['label'].map(label_map)

fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=sample_w,
    x='TEMP',
    y='EDA',
    hue='Condition',
    palette=palette,
    alpha=0.4,
    s=12,
    ax=ax
)

ax.set_title('Wrist: EDA vs TEMP — Raw (Train Split)')

plt.tight_layout()
plt.savefig("wesad_wrist_raw_scatter.png", dpi=150)
plt.show()

print("Saved: wesad_wrist_raw_scatter.png")

# -------------------------------------------------------
# 3. WRIST — rolling window features (1-min and 3-min)
# -------------------------------------------------------
train_engineered = engineer_features(train_df)
train_engineered['Condition'] = train_engineered['label'].map(label_map)

sample_e = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in train_engineered.groupby('label')
]).reset_index(drop=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Wrist: Rolling Window Features by Condition', fontsize=13)

plot_pairs = [
    ('EDA_roll1min_mean', 'EDA_roll1min_std',  'EDA — 1-min window'),
    ('EDA_roll3min_mean', 'EDA_roll3min_std',  'EDA — 3-min window'),
    ('TEMP_roll1min_mean','TEMP_roll1min_std', 'TEMP — 1-min window'),
    ('TEMP_roll3min_mean','TEMP_roll3min_std', 'TEMP — 3-min window'),
]

for ax, (xcol, ycol, title) in zip(axes.flatten(), plot_pairs):
    sns.scatterplot(
        data=sample_e, x=xcol, y=ycol,
        hue='Condition', palette=palette,
        alpha=0.4, s=12, ax=ax, legend=(ax == axes[0][0])
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Rolling Mean', fontsize=9)
    ax.set_ylabel('Rolling Std', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

axes[0][0].legend(title='Condition', fontsize=8, title_fontsize=9)
plt.tight_layout()
plt.savefig("wesad_wrist_rolling_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wesad_wrist_rolling_scatter.png")