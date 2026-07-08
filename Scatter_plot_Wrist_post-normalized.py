import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# Load raw wrist for left plot
wrist = pd.read_csv("wesad_wrist_clean.csv")

label_map = {0: 'Non-Stress', 1: 'Stress'}
palette   = {'Non-Stress': '#378ADD', 'Stress': '#D85A30'}

# -------------------------------------------------------
# POST-NORMALIZATION SCATTER — Wrist (after StandardScaler)
# -------------------------------------------------------
train_normalized = pd.read_csv("wesad_wrist_train.csv")

sample_n = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in train_normalized.groupby('label')
]).reset_index(drop=True)
sample_n['Condition'] = sample_n['label'].map(label_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — raw wrist (pre-normalization)
sample_raw = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in wrist.groupby('label')
]).reset_index(drop=True)
sample_raw['Condition'] = sample_raw['label'].map(label_map)

sns.scatterplot(data=sample_raw, x='TEMP', y='EDA',
                hue='Condition', palette=palette,
                alpha=0.4, s=12, ax=axes[0])
axes[0].set_title('Wrist: EDA vs TEMP — Raw (Pre-normalization)', fontsize=11)
axes[0].set_xlabel('TEMP (°C)')
axes[0].set_ylabel('EDA (μS)')
axes[0].spines[['top', 'right']].set_visible(False)

# Right — normalized wrist (post StandardScaler)
sns.scatterplot(data=sample_n, x='TEMP', y='EDA',
                hue='Condition', palette=palette,
                alpha=0.4, s=12, ax=axes[1])
axes[1].set_title('Wrist: EDA vs TEMP — Post-normalization (StandardScaler)', fontsize=11)
axes[1].set_xlabel('TEMP (z-score)')
axes[1].set_ylabel('EDA (z-score)')
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('Wrist Signals — Effect of Normalization', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("wesad_wrist_normalization_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wesad_wrist_normalization_comparison.png")