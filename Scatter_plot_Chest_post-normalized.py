import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# Load raw chest for left plot
chest = pd.read_csv("wesad_chest_clean.csv")

# Keep only Baseline and Stress
chest = chest[chest["label"].isin([1, 2])].copy()

label_map = {
    1: "Baseline",
    2: "Stress"
}

palette = {
    "Baseline": "#378ADD",
    "Stress": "#D85A30"
}

# -------------------------------------------------------
# POST-NORMALIZATION SCATTER — chest (after StandardScaler)
# -------------------------------------------------------
train_normalized = pd.read_csv("wesad_chest_train.csv")

sample_n = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in train_normalized.groupby('label')
]).reset_index(drop=True)
sample_n['Condition'] = sample_n['label'].map(label_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left — raw chest (pre-normalization)
sample_raw = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in chest.groupby('label')
]).reset_index(drop=True)
sample_raw['Condition'] = sample_raw['label'].map(label_map)

sns.scatterplot(data=sample_raw, x='TEMP', y='EDA',
                hue='Condition', palette=palette,
                alpha=0.4, s=12, ax=axes[0])
axes[0].set_title('Chest: EDA vs TEMP — Raw (Pre-normalization)', fontsize=11)
axes[0].set_xlabel('TEMP (°C)')
axes[0].set_ylabel('EDA (μS)')
axes[0].spines[['top', 'right']].set_visible(False)

# Right — normalized chest (post StandardScaler)
sns.scatterplot(data=sample_n, x='TEMP', y='EDA',
                hue='Condition', palette=palette,
                alpha=0.4, s=12, ax=axes[1])
axes[1].set_title('Chest: EDA vs TEMP — Post-normalization (StandardScaler)', fontsize=11)
axes[1].set_xlabel('TEMP (z-score)')
axes[1].set_ylabel('EDA (z-score)')
axes[1].spines[['top', 'right']].set_visible(False)

plt.suptitle('Chest Signals — Effect of Normalization', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("wesad_chest_normalization_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wesad_chest_normalization_comparison.png")