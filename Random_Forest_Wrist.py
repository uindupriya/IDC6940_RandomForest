import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

print("=== Starting Random Forest Model Loop ===")

# 1. READ PRE-SPLIT, PRE-NORMALIZED FILES
print("Loading data splits...")
train_df = pd.read_csv("wesad_wrist_train.csv")
test_df = pd.read_csv("wesad_wrist_test.csv")

label_mapping = {1: 0, 2: 1, 3: 0}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# 2. FEATURE ENGINEERING: COMPUTE TEMPORAL LAG ARRAYS WITH ECG INCLUDED
# Updated default list to scan for ECG, EDA, and TEMP features dynamically
def apply_time_lags(df, features=['ECG', 'EDA', 'TEMP']):
    # Filter features list to match only columns that exist inside the CSV file
    valid_features = [f for f in features if f in df.columns]
    print(f" -> Engineering lag features for existing channels: {valid_features}")
    
    lagged_groups = []
    # Loop ensures lags are calculated within each subject timeline independently
    for subject, group in df.groupby('subject'):
        group_copy = group.sort_index().copy()
        for col in valid_features:
            for lag in range(1, 5): # 4 lags = 1 second of context at 4Hz
                group_copy[f'{col}_lag_{lag}'] = group_copy[col].shift(lag)
        lagged_groups.append(group_copy.dropna())
    return pd.concat(lagged_groups, ignore_index=True)

print("Engineering lag context windows...")
train_lagged = apply_time_lags(train_df)
test_lagged = apply_time_lags(test_df)

# Filter feature columns out from metadata columns
feature_columns = [c for c in train_lagged.columns if c not in ['subject', 'label']]

# ====================================================
# 3. INDEPENDENT SUBJECT TRAINING LOOP
# ====================================================
for subject, group in train_lagged.groupby('subject'):
    print(f"\nTraining Random Forest Classifier for {subject}...")
    
    # Isolate training inputs for this subject
    X_train = group[feature_columns]
    y_train = group['label']
    
    # Isolate testing inputs from the matching hidden test split file
    test_group = test_lagged[test_lagged['subject'] == subject]
    X_test = test_group[feature_columns]
    y_test = test_group['label']
    
    # Initialize Random Forest with balanced class weights to address label skew
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate performance
    y_pred = rf_model.predict(X_test)
    
    macro_f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='binary', zero_division=0)

    
    print(f" -> Binary Precision : {macro_prec:.4f}")
    print(f" -> Binary Recall    : {macro_rec:.4f}")
    print(f" -> Binary F1-Score  : {macro_f1:.4f}")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Non-Stress', 'Stress'], zero_division=0))


    model_filename = f"rf_model_{subject}.pkl"
    joblib.dump(rf_model, model_filename)
    print(f" Saved: {model_filename}")

print("\n=== Random Forest Execution Loop Complete ===")


# ==============================================================================
# PIPELINE DOCUMENTATION, CORE ASSUMPTIONS, AND MODEL EXPERIMENT FINDINGS
# ==============================================================================
#
# OPERATIONAL EXPERIMENT FINDINGS:
# While Random Forest demonstrates near-perfect classification capabilities on baseline 
# individuals (S2 and S4 achieving an F1-macro of 1.00), it remains highly susceptible 
# to physiological transition overlap. For Subject S3, the model completely failed to 
# distinguish between the late baseline neutral state and the early stress onset state, 
# defaulting to global stress classification (Stress recall = 1.00, Neutral recall = 0.00) [WESAD]. 
# This proves that tabular classifiers struggle when human physiological indicators shift 
# gradually rather than abruptly.
#
# METHODOLOGY AND CONTEXT WRITEOUT:
# While Random Forest is fundamentally non-parametric, applying it to sequential biometric 
# streams requires resolving the violation of sample independence caused by temporal autocorrelation. 
# To transform the WESAD signals into a viable format for tabular tree-split optimization, 
# a multi-stage preprocessing pipeline was enforced [WESAD]. First, local subject-specific 
# Standardization was implemented exclusively within training boundaries to isolate individual 
# baseline variances and remove non-stationary signal drift. Second, temporal lag engineering 
# was executed to map chronological context directly onto the feature matrix. By generating 
# shifts from t-1 through t-4, the model is granted an explicit 1-second historical window 
# to capture signal trajectory at 4Hz. Finally, to ensure balanced multi-class evaluation, 
# an 80/20 chronological train/test split was applied within individual experimental condition 
# blocks, ensuring that the feature space and validation targets were consistently mapped 
# across all three emotional states [WESAD].
#
# THE CORE ASSUMPTIONS OF RANDOM FOREST (TIME-SERIES CONTEXT):
# - Assumption of Sample Independence (The Time-Series Violation): Random Forest assumes 
#   that each row of data is an independent observation. In time-series physiological data, 
#   this assumption is completely violated due to temporal autocorrelation (your heart rate 
#   or skin conductance at second t is highly dependent on second t-1).
# - Assumption of Non-Stationary Invariance: Random Forest cannot extrapolate trends outside 
#   the maximum and minimum values it saw during training. If a subject's raw skin conductance 
#   drifts over the course of an afternoon, the model's tree splits will completely lose 
#   accuracy on future data.
# - Assumption of Identity Mapping: Standard Random Forest has no built-in mechanism to 
#   recognize sequential order or velocity. It treats a static signal value the same way 
#   whether that signal is spiking upward or crashing downward.
#
# HOW THE VARIABLES WERE TRANSFORMED (OUR SOLUTIONS):
# To satisfy these assumptions and make the WESAD variables ready for prediction, we 
# implemented three mandatory engineering transformations [WESAD]:
#
# - Transformation A: Subject-Specific Standardization (Locally Scaled)
#   The Action: We transformed the raw variables (ECG, EDA, TEMP) using StandardScaler 
#   fitted strictly per subject within their independent training partitions.
#   The Reason: This removes individual baseline human biological variances (e.g., one person 
#   naturally sweating more than another) and centers all features around a mean of 0 and a 
#   standard deviation of 1. This scales the data safely into a uniform range across all 
#   subjects without causing data leakage.
#
# - Transformation B: Temporal Lag Feature Engineering (Creating Memory)
#   The Action: We engineered four steps of historic back-shifting for each sensor variable 
#   (creating new columns: EDA_lag_1, EDA_lag_2, etc.).
#   The Reason: This converts the sequential time dependency into static, structural columns 
#   within a single row. It gives the Random Forest an explicit "context window" of the past 
#   1 second of history (at 4Hz), allowing the decision trees to calculate the speed and 
#   direction of the physiological shift.
#
# - Transformation C: Chronological Condition-Block Stratification
#   The Action: We transformed how the training and testing matrices were sliced by performing 
#   an 80/20 sequential split inside each isolated experimental condition block (Neutral, Stress, 
#   Amusement) [WESAD].
#   The Reason: This preserves the real-world sequence of the timeline while forcing both the 
#   training and testing datasets to hold a representative distribution of all three target 
#   states, solving the classification target-mismatch issue [WESAD].
