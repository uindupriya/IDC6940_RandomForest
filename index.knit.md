---
title: "Stress State Prediction from Wearable Sensors: A Comparative Study of ARIMA and Random Forest on the WESAD Dataset"

author: "Indupriya Uppaluri, Virginia Vaughan, Chawan Sreelekha (Advisor: Dr. Cohen)"
date: '2026-06-27'
format:
  html:
    code-fold: true
course: Capstone Projects in Data Science
bibliography: references.bib # file contains bibtex for references
#always_allow_html: true # this allows to get PDF with HTML features
self-contained: true
execute: 
  warning: false
  message: false
editor: 
  markdown: 
    wrap: 72
---

Slides: [slides.html](slides.html){target="_blank"} ( Go to `slides.qmd`
to edit)



## Introduction


Stress detection in real-time, and more importantly biomarkers, have gained interest as both military and corporate facilities aim to improve the quality of worker health and performance. This has let to extensive research using commercial wearables that record signals such as beats per minute (BPM), heartrate (HR), heart rate variability (HRV), galvanic skin conductance (EDA), respiration (RESP), and temperature (TEMP). These physiological signals are used in tandem with both stochastic models such as ARIMA and machine learning such as Random Forest. Both types of models offer their own positives and negatives when predicting physiological data. For example, ARIMA is a more traditional, linear-based model, requires less data for training and extrapolation, while Random Forest is more data-intensive but does not assume linearity, making it better suited for modeling complex patterns. 

In this paper, we hope to explore whether it is possible for the simpler ARIMA model to predict stress state at the same accuracy of a machine learning model like Random Forest. If the ARIMA can perform similar to or greater, this makes an argument that it is still possible to predict stress in physiological signals without the intensive training needed in machine learning, despite ARIMA’s assumptions and limitations. To explore this, we chose the WESAD dataset, which contains multimodal physiological data collected from participants as they underwent different activities, such as no-stress, stress, and amusement. Based on our literature review, we have found that the best physiological signals for predicting stress state that are also provided by the dataset include HR, HRV, and EDA, and that these metrics can be enhanced with secondary metrics such as RESP and TEMP. One paper, [@oyeleye2022] found that the accuracy of ARIMA prediction can be heavily improved using the “rolling window” approach. This helped the model predict based on the signal during different windows of time instead of treating each sample independently of the other. In our methodology we plan to create a pipeline to preprocess the data, compute our feature metrics, derive rolling windows, and predict a given metric using both ARIMA and Random Forest in order to compare the model’s accuracy. 


## Literature review


ARIMA, which stands for Autoregressive Integrated Moving Average, is a classical statistical model widely used for time-series forecasting that captures temporal dependencies within a signal using its own past values and error terms. [@kontopoulou2023] found that ARIMA remains a broadly used and interpretable approach, particularly well-suited for structured, lower-dimensional datasets where its linearity assumptions hold reasonably well. Critically, [@ziyadidegan2025] identified ARIMA as an underutilized but promising method specifically for physiological stress research, appearing in only 3 of 119 reviewed cardiovascular stress studies despite its recognized capacity to capture temporal rhythms in cardiovascular and electrodermal signals. Given that ECG-derived metrics reflect the electrical activity of the heart and EDA captures the skin's electrodermal response to psychological arousal, both of which change measurably during stress, the temporal structure of these signals makes them theoretically well-suited for time-series modeling through ARIMA with a rolling window approach. If stress-related changes in ECG and EDA follow predictable temporal patterns, ARIMA may be capable of detecting stress states without the computational demands of machine learning, a question this study directly investigates. 

A systematic review of the literature on the classification of multimodal sympathetic and parasympathetic nervous system arousal showed that EDA, a measure of sympathetic nervous system activity, and HRV, a measure of parasympathetic nervous system activity, are a more complete reconstruction of autonomic state than either signal alone. We therefore chose respiration as our indicator of stress because it provides an additional measurement of the changes in breathing pattern that occur when a person is stressed, as well as the electrodermal response measured by EDA.

Random Forest is an ensemble machine learning method that builds multiple decision trees and aggregates their outputs to produce robust classifications, making no assumptions about the linearity or distribution of the underlying data. [@garg2021] evaluated Random Forest alongside several other classifiers including k-NN, Linear Discriminant Analysis, AdaBoost, and Support Vector Machine on the WESAD dataset, finding that Random Forest achieved consistently strong performance across both binary stress versus non-stress and three-class neutral, stress, and amusement classification tasks. [@schmidt2018], who originally introduced the WESAD dataset, a multimodal physiological signal collection from 15 subjects recorded across controlled stress, baseline, and amusement conditions using wrist-worn and chest-worn devices capturing signals including ECG and EDA among others, also conducted preliminary Random Forest experiments achieving 88.33% baseline accuracy, further establishing it as the benchmark comparator for this study. Together, these findings position Random Forest as a well-validated and high-performing method for stress classification on WESAD, and the central question of this study is whether the simpler, interpretable ARIMA model can achieve comparable predictive performance on ECG and EDA signals alone.


## Methods

## Analysis



### Random Forest Classification Framework

The predictive model used for multi-class affective state classification is the non-parametric Random Forest algorithm. Formally, a Random Forest is an ensemble classifier consisting of a collection of structured decision trees $\{h(x, \theta_k), k = 1, \dots\}$ where the parameters $\{\theta_k\}$ are independent, identically distributed random vectors, and each individual tree casts a single unit vote for the most popular class at input feature vector $x$.

The foundational non-parametric regression and classification framework can be defined by the following sequence:

$$Y_i = m(X_i) + \varepsilon_i$$

where $Y_i$ represents the target affective stress label ($Y_i \in \{1, 2, 3\}$: Neutral, Stress, Amusement), which is defined as the sum of the true underlying physiological mapping function value $m(x)$ for the feature tensor $X_i$, and $\varepsilon_i$ represents the random observation errors.

In this implementation, the target function $m(x)$ is mathematically unknown. With the help of this definition, we create a robust local averaging estimation by aggregating a multitude of decorrelated randomized decision trees ($T$). The ultimate classification function estimation formula is printed below:

$$\hat{m}(x) = \text{mode} \left\{ T_1(x), T_2(x), \dots, T_B(x) \right\}$$

In other words, this means that we are discovering the emotional state boundaries across the physiological multi-sensor space with the help of bootstrap aggregation (bagging) and randomized feature sub-selection across $B$ independent trees ($B = 100$). This collective voting structure decreases the overall structural variance of individual decision trees without increasing the model's bias.

#### Justification of Choices Based on Problem and Data
We chose the Random Forest algorithm for this comparative time-series framework based on specific properties of the WESAD wearable dataset:

- **Handling of Non-Linear Physiological Interaction:** Autonomic nervous system responses do not follow strict linear bounds; a spike in skin conductance (EDA) combined with a drop in skin temperature (TEMP) creates highly complex, non-linear patterns during stress episodes.
- **Robustness to Sensor Artifacts and Noise:** Wearable sensor data collected in the field is prone to sudden movement artifacts. Because Random Forest splits nodes by evaluating localized subsets of data and features via the Gini impurity index, it prevents individual extreme outliers or sensor errors from corrupting the global classification boundaries.

### Assumptions of Random Forest

## OPERATIONAL EXPERIMENT FINDINGS:
  While Random Forest demonstrates near-perfect classification capabilities on baseline individuals (S2 and S4 achieving an F1-macro of 1.00), it remains highly susceptible to physiological transition overlap. For Subject S3, the model completely failed to distinguish between the late baseline neutral state and the early stress onset state, defaulting to global stress classification (Stress recall = 1.00, Neutral recall = 0.00[@schmidt2018]. This proves that tabular classifiers struggle when human physiological indicators shift gradually rather than abruptly.


## THE CORE ASSUMPTIONS OF RANDOM FOREST (TIME-SERIES CONTEXT):
  - Assumption of Sample Independence (The Time-Series Violation): Random Forest assumes 
    that each row of data is an independent observation. In time-series physiological data, 
    this assumption is completely violated due to temporal autocorrelation (your heart rate 
    or skin conductance at second t is highly dependent on second t-1).
  - Assumption of Non-Stationary Invariance: Random Forest cannot extrapolate trends outside 
    the maximum and minimum values it saw during training. If a subject's raw skin conductance 
    drifts over the course of an afternoon, the model's tree splits will completely lose 
    accuracy on future data.
  - Assumption of Identity Mapping: Standard Random Forest has no built-in mechanism to 
    recognize sequential order or velocity. It treats a static signal value the same way 
    whether that signal is spiking upward or crashing downward.

  ### HOW THE VARIABLES WERE TRANSFORMED (OUR SOLUTIONS):
  To satisfy these assumptions and make the WESAD variables ready for prediction, we 
  implemented three mandatory engineering transformations [@garg2021]:
 
  - Transformation A: Subject-Specific Standardization (Locally Scaled)
    The Action: We transformed the raw variables (ECG, EDA, TEMP) using StandardScaler 
    fitted strictly per subject within their independent training partitions.
    The Reason: This removes individual baseline human biological variances (e.g., one person 
    naturally sweating more than another) and centers all features around a mean of 0 and a 
    standard deviation of 1. This scales the data safely into a uniform range across all 
    subjects without causing data leakage.
 
  - Transformation B: Temporal Lag Feature Engineering (Creating Memory)
    The Action: We engineered four steps of historic back-shifting for each sensor variable 
    (creating new columns: EDA_lag_1, EDA_lag_2, etc.).
    The Reason: This converts the sequential time dependency into static, structural columns 
    within a single row. It gives the Random Forest an explicit "context window" of the past 
    1 second of history (at 4Hz), allowing the decision trees to calculate the speed and 
    direction of the physiological shift.
 
  - Transformation C: Chronological Condition-Block Stratification
    The Action: We transformed how the training and testing matrices were sliced by performing 
    an 80/20 sequential split inside each isolated experimental condition block (Neutral, Stress, 
    Amusement) [@schmidt2018].
    The Reason: This preserves the real-world sequence of the timeline while forcing both the 
    training and testing datasets to hold a representative distribution of all three target 
    states, solving the classification target-mismatch issue [@kontopoulou2023].

$$ M_n(x) = \sum_{i=1}^{n} W_n (X_i) Y_i \tag{1} $$

*$W_n(x)$ is the sum of weights that belongs to all real numbers. Weights are positive numbers and small if $X_i$ is far from $x$.*

#### Justification of Choices Based on Problem and Data
This non-parametric local averaging framework (Equation 1) is explicitly chosen for our WESAD wearable sensor prediction task instead of standard parametric regression algorithms, which typically assume a strict linear profile [@schmidt2018]:

$$ y_i = \beta_0 + \beta_1 X_1 +\varepsilon_i \tag{2} $$

A parametric linear model (Equation 2) assumes a constant, straight-line relationship that completely fails when applied to the complex, non-linear biological response profiles found within multi-modal wearable sensor streams [@ziyadidegan2025]. We chose the Random Forest framework based on three specific data properties that align with this non-parametric local estimation approach:

- **Handling of Non-Linear Physiological Interaction:** Autonomic nervous system responses do not follow strict linear bounds; a spike in skin conductance (EDA) combined with a drop in skin temperature (TEMP) creates highly complex, non-linear patterns during stress episodes [@ziyadidegan2025]. Random Forest handles these multi-modal, non-linear relationships out of the box through step-wise feature space partitioning without requiring restrictive linear coefficients like those in Equation 2.
- **Robustness to Sensor Artifacts and Noise:** Wearable sensor data collected in the field is prone to sudden movement artifacts and temporary data dropouts [@garg2021]. Because Random Forest splits nodes by evaluating localized subsets of data and features via the Gini impurity index, it acts as a robust local estimator, preventing individual extreme outliers or sensor errors from corrupting the global classification boundaries.
- **Resistance to Temporal Overfitting:** High-frequency biological streams exhibit intense temporal autocorrelation, which can cause standard models to overfit on specific timeline segments. By utilizing independent bootstrap samples for each tree and forcing a structural 80/20 chronological split inside isolated condition blocks, the Random Forest maintains strict out-of-sample operational testing validity [@kontopoulou2023].



#### Normalization

Prior to model training, raw physiological signals (ECG, EDA) were normalized using subject-specific StandardScaler to remove individual baseline biological variance. Each subject's training partition was fitted independently to prevent data leakage across subjects. The normalization formula applied is:

$$z = \frac{x - \mu}{\sigma}$$

where $x$ is the raw signal value, $\mu$ is the subject-specific mean, 
and $\sigma$ is the subject-specific standard deviation computed strictly 
within the training partition [@schmidt2018].

#### Rolling Window Features

To capture the temporal dynamics of physiological stress responses, rolling window features were engineered from the normalized ECG and EDA signals. For each signal at time $t$, a window of size $w$ was applied to compute the rolling mean:

$$\mu_{t,w} = \frac{1}{w}\sum_{i=t-w+1}^{t} x_i$$

and rolling standard deviation:

$$\sigma_{t,w} = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^{t}(x_i - \mu_{t,w})^2}$$

Two window sizes were applied which are 1-minute and 3-minute windows inorder to capture both short-term stress onset patterns and longer-term sustained stress responses [@ziyadidegan2025]. These rolling features give the Random Forest model an explicit temporal context window, partially 
compensating for its lack of built-in sequential awareness [@kontopoulou2023].

#### Feature Importance

Random Forest provides a built-in feature importance measure based on the mean decrease in impurity across all trees for each feature. This was used to identify which physiological signals and rolling window features most strongly predicted stress state classification across subjects S2, S3, and S4 from the WESAD dataset [@garg2021].

## Data Ingestion and Automated Feature Alignment

The initial phase of the predictive pipeline establishes a robust data ingestion layer designed to load partitioned datasets, dynamically detect available sensor modalities, and enforce structural timeline isolation.
 Ingestion Protocols and Format Mapping - The processing script programmatically interfaces with two pre-arranged disk files: wesad_wrist_train.csv (the training matrix) and wesad_wrist_test.csv (the holdout evaluation matrix). The ingestion sequence is mapped via the pandas framework using explicit memory allocation constraints:Tabular Parsing: The continuous comma-separated streams are read into isolated memory frames (train_df and test_df) as multi-column tabular arrays.DataFrame Casting: Arrays are strictly enforced as traditional mutable matrices (as.data.frame structural equivalents) to allow downstream indexing and chronological row transformations.Random State Initialization: To stabilize any internal stochastic data handling or bootstrapping mechanisms during the initial load phase, a global seed is instantiated using random_state=42. Dynamic Channel Scanning and Feature Verification - Because wearable sensor platforms can experience hardware dropouts, or variations between chest-worn and wrist-worn sensor matrices, the ingestion layer implements a dynamic channel scanner.

The system verifies a default physiological target array consisting of Electrodermal Activity (EDA), Body Temperature (TEMP), and Electrocardiogram (ECG) metrics. Rather than assuming strict structural permanence, the ingestion function runs an inline list comprehension:Valid Features={f∈{ECG, EDA, TEMP}∣f∈CDataFrame}Valid Features equals the set of all f is an element of the set ECG, EDA, TEMP end-set such that f is an element of bold cap C sub DataFrame end-sub end-set

Valid Features={𝑓∈{ECG, EDA, TEMP}∣𝑓∈𝐂DataFrame}

 ∣𝑓∈𝐂DataFrame represents the total set of column strings parsed from the file header. Only features that pass this verification step are pushed into the downstream temporal context matrix, preventing runtime compilation failures if a specific signal modality is missing from the wrist file structure.


 Chronological Grouping and Boundary Defense - To maintain strict data integrity during ingestion and prevent cross-subject timeline pollution, the loading sequence prevents global, un-grouped data operations.Data matrices are partitioned into isolated historical blocks using a split-apply-combine approach on the categorical subject identifier. This ensures that the chronological continuity of each participant's biometric timeline is preserved:

By isolating the timelines within individual subject blocks and explicitly invoking a index-sorting mechanism (.sort_index()), the ingestion layer provides a clear guarantee: no row from Subject A can accidentally inherit historical lag metrics from Subject B.Any boundary artifacts created by tracking changes at the very beginning of a subject's timeline are handled by a .dropna() filter, leaving a clean, continuous, and fully isolated data structure for model training.

### Pre processing


::: {.cell}

```{.python .cell-code}
import pickle
import numpy as np
import pandas as pd
import os

subjects = [
    'S2','S3','S4','S5','S6','S7','S8','S9',
    'S10','S11','S13','S14','S15','S16','S17'
]
VALID_LABELS = [1, 2, 3]
chest_dfs = []
wrist_dfs = []

print("Starting WESAD preprocessing pipeline...")
```

::: {.cell-output .cell-output-stdout}

```
Starting WESAD preprocessing pipeline...
```


:::

```{.python .cell-code}
# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
for subject in subjects:
    path = f'/home/iu6/IDC6940_RandomForest/Dataset_WESAD/{subject}/{subject}.pkl'
    if not os.path.exists(path):
        print(f"[SKIP] {subject} not found")
        continue
        
    print(f"\nProcessing {subject}...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    # Get the raw, unbroken 700Hz timeline bounds
    raw_labels = np.array(data['label']).flatten()
    raw_ecg = np.array(data['signal']['chest']['ECG']).flatten()
    raw_eda = np.array(data['signal']['chest']['EDA']).flatten()
    raw_temp = np.array(data['signal']['chest']['Temp']).flatten()
    raw_resp = np.array(data['signal']['chest']['Resp']).flatten()
    
    # Track minimum length to trim trailing-end sensor shutoff mismatches safely
    min_len = min(len(raw_labels), len(raw_ecg), len(raw_eda), len(raw_temp), len(raw_resp))

    # -------------------------
    # CHEST SIGNALS (100% continuous for NeuroKit2 windowing)
    # -------------------------
    chest_df = pd.DataFrame({
        'subject': subject,
        'ECG': raw_ecg[:min_len],
        'EDA': raw_eda[:min_len],
        'TEMP': raw_temp[:min_len],
        'RESP': raw_resp[:min_len],
        'label': raw_labels[:min_len]  # Keeps ALL labels (0,1,2,3,4...) so data stays unbroken
    })
    
    # Handle artifacts without changing the shape or deleting rows
    chest_df.loc[(chest_df['TEMP'] <= 20) | (chest_df['TEMP'] >= 45), 'TEMP'] = np.nan
    chest_df.loc[chest_df['EDA'] < 0, 'EDA'] = np.nan
    chest_df = chest_df.ffill().bfill()  # Cleans sensor errors seamlessly
    chest_df = chest_df[chest_df['label'].isin(VALID_LABELS)]

    
    chest_dfs.append(chest_df)
    print(f"[CHEST CONTINUOUS] {subject}: {chest_df.shape}")

    # -------------------------
    # WRIST SIGNALS (Downsampled labels to match native 4Hz sensors)
    # -------------------------
    wrist_eda = np.array(data['signal']['wrist']['EDA']).flatten()
    wrist_temp = np.array(data['signal']['wrist']['TEMP']).flatten()
    
    
    wrist_len = min(len(wrist_eda), len(wrist_temp))
    # Pick every 175th label to match the slow 4Hz wrist sensors perfectly by time
    wrist_labels = np.array([np.bincount(raw_labels[i:i+175]).argmax() for i in range(0, len(raw_labels), 175)])[:wrist_len]
    
    wrist_df = pd.DataFrame({
        'subject': subject,
        'EDA': wrist_eda[:wrist_len],
        'TEMP': wrist_temp[:wrist_len],
        'label': wrist_labels
    })
    
    # Wrist can be filtered row-by-row because her NeuroKit pipeline isn't analyzing the wrist
    wrist_df = wrist_df[wrist_df['label'].isin(VALID_LABELS)]
    wrist_df = wrist_df.dropna()
    wrist_dfs.append(wrist_df)
    print(f"[WRIST CLEAN] {subject}: {wrist_df.shape}")
```

::: {.cell-output .cell-output-stdout}

```

Processing S2...
[CHEST CONTINUOUS] S2: (1484700, 6)
[WRIST CLEAN] S2: (8484, 4)

Processing S3...
[CHEST CONTINUOUS] S3: (1508500, 6)
[WRIST CLEAN] S3: (8620, 4)

Processing S4...
[CHEST CONTINUOUS] S4: (1515501, 6)
[WRIST CLEAN] S4: (8660, 4)
[SKIP] S5 not found
[SKIP] S6 not found
[SKIP] S7 not found
[SKIP] S8 not found
[SKIP] S9 not found
[SKIP] S10 not found
[SKIP] S11 not found
[SKIP] S13 not found
[SKIP] S14 not found
[SKIP] S15 not found
[SKIP] S16 not found
[SKIP] S17 not found
```


:::

```{.python .cell-code}
# ----------------------------------------------------
# FINAL DATASET EXPORT
# ----------------------------------------------------
print("\nCombining datasets...")
```

::: {.cell-output .cell-output-stdout}

```

Combining datasets...
```


:::

```{.python .cell-code}
chest_final = pd.concat(chest_dfs, ignore_index=True)
wrist_final = pd.concat(wrist_dfs, ignore_index=True)

print("\nFinal Chest shape (Continuous):", chest_final.shape)
```

::: {.cell-output .cell-output-stdout}

```

Final Chest shape (Continuous): (4508701, 6)
```


:::

```{.python .cell-code}
print("Final Wrist shape (Clean Tabular):", wrist_final.shape)
```

::: {.cell-output .cell-output-stdout}

```
Final Wrist shape (Clean Tabular): (25764, 4)
```


:::

```{.python .cell-code}
chest_final.to_csv("wesad_chest_clean.csv", index=False)
wrist_final.to_csv("wesad_wrist_clean.csv", index=False)
print("\nSaved cleaned datasets successfully. Ready for handoff!")
```

::: {.cell-output .cell-output-stdout}

```

Saved cleaned datasets successfully. Ready for handoff!
```


:::
:::

#| code-fold: true



::: {.cell}

```{.python .cell-code}
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
```
:::

#| code-fold: true


::: {.cell}

```{.python .cell-code}
# Load Data
print("=== Starting Random Forest Model Loop ===")
```

::: {.cell-output .cell-output-stdout}

```
=== Starting Random Forest Model Loop ===
```


:::

```{.python .cell-code}
# 1. READ PRE-SPLIT, PRE-NORMALIZED FILES
print("Loading data splits...")
```

::: {.cell-output .cell-output-stdout}

```
Loading data splits...
```


:::

```{.python .cell-code}
train_df = pd.read_csv("wesad_wrist_train.csv")
test_df = pd.read_csv("wesad_wrist_test.csv")

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
```

::: {.cell-output .cell-output-stdout}

```
Engineering lag context windows...
```


:::

```{.python .cell-code}
train_lagged = apply_time_lags(train_df)
```

::: {.cell-output .cell-output-stdout}

```
 -> Engineering lag features for existing channels: ['EDA', 'TEMP']
```


:::

```{.python .cell-code}
test_lagged = apply_time_lags(test_df)
```

::: {.cell-output .cell-output-stdout}

```
 -> Engineering lag features for existing channels: ['EDA', 'TEMP']
```


:::

```{.python .cell-code}
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
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"[{subject} Metrics Summary]")
    print(f" -> Macro Precision : {macro_prec:.4f}")
    print(f" -> Macro Recall    : {macro_rec:.4f}")
    print(f" -> Macro F1-Score  : {macro_f1:.4f}")
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, labels=[1, 2, 3], target_names=['Neutral', 'Stress', 'Amusement'], zero_division=0))
```

::: {.cell-output-display}

```{=html}
<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

#sk-container-id-1.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

#sk-container-id-1.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(class_weight=&#x27;balanced&#x27;, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>
                    
        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=n_estimators,-int%2C%20default%3D100">
            n_estimators
            <span class="param-doc-description">n_estimators: int, default=100<br><br>The number of trees in the forest.<br><br>.. versionchanged:: 0.22<br>   The default value of ``n_estimators`` changed from 10 to 100<br>   in 0.22.</span>
        </a>
    </td>
            <td class="value">100</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=criterion,-%7B%22gini%22%2C%20%22entropy%22%2C%20%22log_loss%22%7D%2C%20default%3D%22gini%22">
            criterion
            <span class="param-doc-description">criterion: {"gini", "entropy", "log_loss"}, default="gini"<br><br>The function to measure the quality of a split. Supported criteria are<br>"gini" for the Gini impurity and "log_loss" and "entropy" both for the<br>Shannon information gain, see :ref:`tree_mathematical_formulation`.<br>Note: This parameter is tree-specific.</span>
        </a>
    </td>
            <td class="value">&#x27;gini&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=max_depth,-int%2C%20default%3DNone">
            max_depth
            <span class="param-doc-description">max_depth: int, default=None<br><br>The maximum depth of the tree. If None, then nodes are expanded until<br>all leaves are pure or until all leaves contain less than<br>min_samples_split samples.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=min_samples_split,-int%20or%20float%2C%20default%3D2">
            min_samples_split
            <span class="param-doc-description">min_samples_split: int or float, default=2<br><br>The minimum number of samples required to split an internal node:<br><br>- If int, then consider `min_samples_split` as the minimum number.<br>- If float, then `min_samples_split` is a fraction and<br>  `ceil(min_samples_split * n_samples)` are the minimum<br>  number of samples for each split.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">2</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=min_samples_leaf,-int%20or%20float%2C%20default%3D1">
            min_samples_leaf
            <span class="param-doc-description">min_samples_leaf: int or float, default=1<br><br>The minimum number of samples required to be at a leaf node.<br>A split point at any depth will only be considered if it leaves at<br>least ``min_samples_leaf`` training samples in each of the left and<br>right branches.  This may have the effect of smoothing the model,<br>especially in regression.<br><br>- If int, then consider `min_samples_leaf` as the minimum number.<br>- If float, then `min_samples_leaf` is a fraction and<br>  `ceil(min_samples_leaf * n_samples)` are the minimum<br>  number of samples for each node.<br><br>.. versionchanged:: 0.18<br>   Added float values for fractions.</span>
        </a>
    </td>
            <td class="value">1</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=min_weight_fraction_leaf,-float%2C%20default%3D0.0">
            min_weight_fraction_leaf
            <span class="param-doc-description">min_weight_fraction_leaf: float, default=0.0<br><br>The minimum weighted fraction of the sum total of weights (of all<br>the input samples) required to be at a leaf node. Samples have<br>equal weight when sample_weight is not provided.</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=max_features,-%7B%22sqrt%22%2C%20%22log2%22%2C%20None%7D%2C%20int%20or%20float%2C%20default%3D%22sqrt%22">
            max_features
            <span class="param-doc-description">max_features: {"sqrt", "log2", None}, int or float, default="sqrt"<br><br>The number of features to consider when looking for the best split:<br><br>- If int, then consider `max_features` features at each split.<br>- If float, then `max_features` is a fraction and<br>  `max(1, int(max_features * n_features_in_))` features are considered at each<br>  split.<br>- If "sqrt", then `max_features=sqrt(n_features)`.<br>- If "log2", then `max_features=log2(n_features)`.<br>- If None, then `max_features=n_features`.<br><br>.. versionchanged:: 1.1<br>    The default of `max_features` changed from `"auto"` to `"sqrt"`.<br><br>Note: the search for a split does not stop until at least one<br>valid partition of the node samples is found, even if it requires to<br>effectively inspect more than ``max_features`` features.</span>
        </a>
    </td>
            <td class="value">&#x27;sqrt&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=max_leaf_nodes,-int%2C%20default%3DNone">
            max_leaf_nodes
            <span class="param-doc-description">max_leaf_nodes: int, default=None<br><br>Grow trees with ``max_leaf_nodes`` in best-first fashion.<br>Best nodes are defined as relative reduction in impurity.<br>If None then unlimited number of leaf nodes.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=min_impurity_decrease,-float%2C%20default%3D0.0">
            min_impurity_decrease
            <span class="param-doc-description">min_impurity_decrease: float, default=0.0<br><br>A node will be split if this split induces a decrease of the impurity<br>greater than or equal to this value.<br><br>The weighted impurity decrease equation is the following::<br><br>    N_t / N * (impurity - N_t_R / N_t * right_impurity<br>                        - N_t_L / N_t * left_impurity)<br><br>where ``N`` is the total number of samples, ``N_t`` is the number of<br>samples at the current node, ``N_t_L`` is the number of samples in the<br>left child, and ``N_t_R`` is the number of samples in the right child.<br><br>``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,<br>if ``sample_weight`` is passed.<br><br>.. versionadded:: 0.19</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=bootstrap,-bool%2C%20default%3DTrue">
            bootstrap
            <span class="param-doc-description">bootstrap: bool, default=True<br><br>Whether bootstrap samples are used when building trees. If False, the<br>whole dataset is used to build each tree.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=oob_score,-bool%20or%20callable%2C%20default%3DFalse">
            oob_score
            <span class="param-doc-description">oob_score: bool or callable, default=False<br><br>Whether to use out-of-bag samples to estimate the generalization score.<br>By default, :func:`~sklearn.metrics.accuracy_score` is used.<br>Provide a callable with signature `metric(y_true, y_pred)` to use a<br>custom metric. Only available if `bootstrap=True`.<br><br>For an illustration of out-of-bag (OOB) error estimation, see the example<br>:ref:`sphx_glr_auto_examples_ensemble_plot_ensemble_oob.py`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description">n_jobs: int, default=None<br><br>The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,<br>:meth:`decision_path` and :meth:`apply` are all parallelized over the<br>trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`<br>context. ``-1`` means using all processors. See :term:`Glossary<br><n_jobs>` for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>
    

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=random_state,-int%2C%20RandomState%20instance%20or%20None%2C%20default%3DNone">
            random_state
            <span class="param-doc-description">random_state: int, RandomState instance or None, default=None<br><br>Controls both the randomness of the bootstrapping of the samples used<br>when building trees (if ``bootstrap=True``) and the sampling of the<br>features to consider when looking for the best split at each node<br>(if ``max_features < n_features``).<br>See :term:`Glossary <random_state>` for details.</span>
        </a>
    </td>
            <td class="value">42</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=verbose,-int%2C%20default%3D0">
            verbose
            <span class="param-doc-description">verbose: int, default=0<br><br>Controls the verbosity when fitting and predicting.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=warm_start,-bool%2C%20default%3DFalse">
            warm_start
            <span class="param-doc-description">warm_start: bool, default=False<br><br>When set to ``True``, reuse the solution of the previous call to fit<br>and add more estimators to the ensemble, otherwise, just fit a whole<br>new forest. See :term:`Glossary <warm_start>` and<br>:ref:`tree_ensemble_warm_start` for details.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>
    

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('class_weight',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=class_weight,-%7B%22balanced%22%2C%20%22balanced_subsample%22%7D%2C%20dict%20or%20list%20of%20dicts%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3DNone">
            class_weight
            <span class="param-doc-description">class_weight: {"balanced", "balanced_subsample"}, dict or list of dicts,             default=None<br><br>Weights associated with classes in the form ``{class_label: weight}``.<br>If not given, all classes are supposed to have weight one. For<br>multi-output problems, a list of dicts can be provided in the same<br>order as the columns of y.<br><br>Note that for multioutput (including multilabel) weights should be<br>defined for each class of every column in its own dict. For example,<br>for four-class multilabel classification weights should be<br>[{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of<br>[{1:1}, {2:5}, {3:1}, {4:1}].<br><br>The "balanced" mode uses the values of y to automatically adjust<br>weights inversely proportional to class frequencies in the input data<br>as ``n_samples / (n_classes * np.bincount(y))``<br><br>The "balanced_subsample" mode is the same as "balanced" except that<br>weights are computed based on the bootstrap sample for every tree<br>grown.<br><br>For multi-output, the weights of each column of y will be multiplied.<br><br>Note that these weights will be multiplied with sample_weight (passed<br>through the fit method) if sample_weight is specified.</span>
        </a>
    </td>
            <td class="value">&#x27;balanced&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=ccp_alpha,-non-negative%20float%2C%20default%3D0.0">
            ccp_alpha
            <span class="param-doc-description">ccp_alpha: non-negative float, default=0.0<br><br>Complexity parameter used for Minimal Cost-Complexity Pruning. The<br>subtree with the largest cost complexity that is smaller than<br>``ccp_alpha`` will be chosen. By default, no pruning is performed. See<br>:ref:`minimal_cost_complexity_pruning` for details. See<br>:ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`<br>for an example of such pruning.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">0.0</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=max_samples,-int%20or%20float%2C%20default%3DNone">
            max_samples
            <span class="param-doc-description">max_samples: int or float, default=None<br><br>If bootstrap is True, the number of samples to draw from X<br>to train each base estimator.<br><br>- If None (default), then draw `X.shape[0]` samples.<br>- If int, then draw `max_samples` samples.<br>- If float, then draw `max(round(n_samples * max_samples), 1)` samples. Thus,<br>  `max_samples` should be in the interval `(0.0, 1.0]`.<br><br>.. versionadded:: 0.22</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html#:~:text=monotonic_cst,-array-like%20of%20int%20of%20shape%20%28n_features%29%2C%20default%3DNone">
            monotonic_cst
            <span class="param-doc-description">monotonic_cst: array-like of int of shape (n_features), default=None<br><br>Indicates the monotonicity constraint to enforce on each feature.<br>  - 1: monotonic increase<br>  - 0: no constraint<br>  - -1: monotonic decrease<br><br>If monotonic_cst is None, no constraints are applied.<br><br>Monotonicity constraints are not supported for:<br>  - multiclass classifications (i.e. when `n_classes > 2`),<br>  - multioutput classifications (i.e. when `n_outputs_ > 1`),<br>  - classifications trained on data with missing values.<br><br>The constraints hold over the probability of the positive class.<br><br>Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.<br><br>.. versionadded:: 1.4</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>
    
                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});


/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-1');</script></body>
```

:::

```{.python .cell-code}
print("\n=== Random Forest Execution Loop Complete ===")
```

::: {.cell-output .cell-output-stdout}

```

=== Random Forest Execution Loop Complete ===
```


:::
:::

### Data and Visualization

::: {.cell}

```{.python .cell-code  code-fold="true"}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

label_map = {1: 'Baseline', 2: 'Stress', 3: 'Amusement'}
palette   = {'Baseline': '#378ADD', 'Stress': '#D85A30', 'Amusement': '#1D9E75'}
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
chest = pd.read_csv("wesad_chest_clean.csv")

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
train_df = pd.read_csv("wesad_wrist_train.csv")

sample_w = pd.concat([
    grp.sample(min(3000, len(grp)), random_state=42)
    for _, grp in train_df.groupby('label')
]).reset_index(drop=True)
sample_w['Condition'] = sample_w['label'].map(label_map)

fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(data=sample_w, x='TEMP', y='EDA',
                hue='Condition', palette=palette,
                alpha=0.4, s=12, ax=ax)
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
```
:::


    Before normalization, raw signal distributions were examined across 
    all three conditions to identify class separation patterns.

    ![WESAD Chest Signals — Pre-normalization](wesad_chest_pairplot.png){width=100%}

![Wrist: EDA vs TEMP — Raw](wesad_wrist_raw_scatter.png){width=70%}

![Wrist: Rolling Window Features by Condition](wesad_wrist_rolling_scatter.png){width=90%}

![WESAD Chest Signals — Pre-normalization](wesad_chest_pairplot.png){width=100%}

![WESAD Chest Signals — Post-normalization](wesad_wrist_normalization_comparison.png){width=100%}

### Modeling and Results

-   Explain your data preprocessing and cleaning steps.

-   Present your key findings in a clear and concise manner.

-   Use visuals to support your claims.

-   **Tell a story about what the data reveals.**


::: {.cell}

:::


### Conclusion

-   Summarize your key findings.

-   Discuss the implications of your results.

## References
Oyeleye M, Chen T, Titarenko S, Antoniou G. A Predictive Analysis of Heart Rates Using Machine Learning Techniques. Int J Environ Res Public Health. 2022 Feb 19;19(4):2417. doi: 10.3390/ijerph19042417. PMID: 35206603; PMCID: PMC8872524. 

Garg P, Santhosh J, Dengel A, Ishimaru S. Stress Detection by Machine Learning and Wearable Sensors. IUI '21 Companion. 2021. doi: 10.1145/3397482.3450732

Schmidt P, Reiss A. WESAD (Wearable Stress and Affect Detection). ACM ICMI 2018. doi: 10.1145/3242969.3242985

Kontopoulou VI, Panagopoulos AD, Kakkos I, Matsopoulos GK. A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks. Future Internet. 2023;15(8):255. doi: 10.3390/fi15080255

Ziyadidegan S, Sadeghi N, Razavi M, Baharlouei E, Janfaza V, Kazeminasab S, Pesarakli H, Javid AH, Sasangohar F. Quantifying Mental Stress Using Cardiovascular Responses: A Scoping Review. 2024. PMC12300869.
