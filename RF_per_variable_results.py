# save per variable RF results to CSV
per_variable_results = [
    {'Signal': 'ECG', 'Subject': 'S2', 'Precision': 0.4295, 'Recall': 0.4292, 'F1': 0.3545},
    {'Signal': 'ECG', 'Subject': 'S3', 'Precision': 0.5828, 'Recall': 0.5989, 'F1': 0.5692},
    {'Signal': 'ECG', 'Subject': 'S4', 'Precision': 0.5613, 'Recall': 0.5645, 'F1': 0.5623},
    {'Signal': 'EDA', 'Subject': 'S2', 'Precision': 0.8273, 'Recall': 0.5123, 'F1': 0.4414},
    {'Signal': 'EDA', 'Subject': 'S3', 'Precision': 0.1976, 'Recall': 0.2823, 'F1': 0.1796},
    {'Signal': 'EDA', 'Subject': 'S4', 'Precision': 1.0000, 'Recall': 1.0000, 'F1': 1.0000},
    {'Signal': 'TEMP', 'Subject': 'S2', 'Precision': 1.0000, 'Recall': 1.0000, 'F1': 1.0000},
    {'Signal': 'TEMP', 'Subject': 'S3', 'Precision': 0.7754, 'Recall': 0.8278, 'F1': 0.7514},
    {'Signal': 'TEMP', 'Subject': 'S4', 'Precision': 0.9993, 'Recall': 0.9997, 'F1': 0.9995},
    {'Signal': 'RESP', 'Subject': 'S2', 'Precision': 0.5281, 'Recall': 0.5324, 'F1': 0.5247},
    {'Signal': 'RESP', 'Subject': 'S3', 'Precision': 0.6411, 'Recall': 0.6400, 'F1': 0.6406},
    {'Signal': 'RESP', 'Subject': 'S4', 'Precision': 0.6220, 'Recall': 0.6395, 'F1': 0.6242},
]

import pandas as pd
df_results = pd.DataFrame(per_variable_results)
df_results.to_csv('RF_Per_Variable_Results.csv', index=False)
print("Saved: RF_Per_Variable_Results.csv")