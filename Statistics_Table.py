import pandas as pd

df = pd.read_csv("wesad_chest_clean.csv")

label_map = {1: 'Neutral', 2: 'Stress', 3: 'Amusement'}
df['Condition'] = df['label'].map(label_map)

stats = df.groupby('Condition')[['ECG','EDA','TEMP','RESP']].agg(['mean','std','min','max']).round(3)
print(stats.to_string())