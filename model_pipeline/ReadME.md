# README.md


# WESAD ARIMA Prediction

A configurable Python pipeline for forecasting physiological stress-related signals from the **WESAD (Wearable Stress and Affect Detection)** dataset using **ARIMA** time-series models.

This project supports:

- Participant-specific or composite participant modeling
- Automatic physiological feature extraction (e.g., ECG → HR/HRV)
- Config-driven experimentation
- Manual ARIMA parameters or AutoARIMA parameter selection
- Continuous forecasting evaluation metrics

---

# Project Overview

The goal of this project is to forecast physiological signals associated with stress using wearable sensor data from the WESAD dataset.

Current workflow:

1. Load WESAD participant data
2. Extract physiological features
3. Create train/test time-series splits
4. Train ARIMA models
5. Forecast physiological signals
6. Evaluate forecasting performance

---

# Dataset

This project uses the **WESAD (Wearable Stress and Affect Detection)** dataset.

Expected structure:

```text
WESAD/
├── S2/
│   └── S2.pkl
├── S3/
│   └── S3.pkl
├── S4/
│   └── S4.pkl
```
---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd WESAD_ARIMA_Prediction
````

## 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run with default config:

```bash
python main.py
```

Run with specific config:

```bash
python main.py config.yaml
```

---

# Configuration File

Pipeline behavior is controlled using `config.yaml`.

Example:

```yaml
# Absolute WESAD dataset path
dataset_path: 'C:\Users\myusername\WESAD\'

# Processing
multithread: false

# Participants
composite_participants: false
participants: ["S2", "S3"]

# Variables + feature extraction
variables:
  ECG: ["HR", "HRV"]

# ARIMA Parameters
# If ALL are None → AutoARIMA
p: None
d: None
q: None
```

---

## Config Parameters

### dataset_path

Absolute path to the WESAD dataset.

Example:

```yaml
dataset_path: 'C:\Users\vgvau\WESAD\'
```

---

### multithread

Enable multithreaded participant processing.

```yaml
multithread: false
```

Current status: **not yet implemented**

---

### composite_participants

Determines training strategy.

#### false (Recommended)

Each participant receives their own model.

Example:

```yaml
composite_participants: false
participants: ["S2", "S3"]
```

Result:

* S2 → individual ARIMA model
* S3 → individual ARIMA model

Recommended for **participant-specific physiological modeling**.

Combines all participant datapoints into one dataset before training.

Example:

```yaml
composite_participants: true
```

Limitations:

* physiology differs between participants
* may reduce personalization
* possible participant leakage

---

### participants

List of WESAD participants to process.

Example:

```yaml
participants: ["S2", "S3"]
```

---

### variables

Defines:

1. Which raw physiological signal to load
2. Which derived features to compute

Example:

```yaml
variables:
  ECG: ["HR", "HRV"]
```

Meaning ECG signal is used to derive HR and HRV

---

# Feature Maps

Current supported signals:

## ECG

Raw electrocardiogram waveform.

Derived features:

| Feature | Description            |
| ------- | ---------------------- |
| HR      | Heart Rate (BPM)       |
| HRV     | Heart Rate Variability |

Example:

```yaml
variables:
  ECG: ["HR", "HRV"]
```

---

# Time-Series Splitting

Default split:

* 70% Train
* 30% Test

Time order is preserved.

No shuffling is performed:

```python
shuffle=False
```

This prevents future data leakage.

---

# ARIMA Configuration

## Manual Parameters

Specify ARIMA order manually.

Example:

```yaml
p: 2
d: 1
q: 3
```

Runs:

```text
ARIMA(2,1,3)
```

---

## AutoARIMA

If ALL parameters are None:

```yaml
p: None
d: None
q: None
```

The pipeline automatically finds the best order.

Example output:

```text
Using AutoARIMA...
Selected Order: (2,1,1)
```

---

# Evaluation Metrics

Continuous forecasting metrics:

| Metric | Description                    |
| ------ | ------------------------------ |
| MAE    | Mean Absolute Error            |
| MSE    | Mean Squared Error             |
| RMSE   | Root Mean Squared Error        |
| MAPE   | Mean Absolute Percentage Error |

Example output:

```text
Participant: S2
ARIMA Order: (2,1,1)

MAE: 2.11
MSE: 7.42
RMSE: 2.72
MAPE: 5.81%
```

---

# WESAD Notes

## ECG is NOT Heart Rate

ECG is the raw electrical waveform.

Example ECG values:

```text
-0.021
-0.018
-0.015
```

These are expected.

Pipeline:

```text
ECG
 ↓
R Peak Detection
 ↓
Heart Rate (HR)
```

---

## Sampling Rates

| Signal | Sampling Rate |
| ------ | ------------- |
| ECG    | 700 Hz        |
| Resp   | 700 Hz        |
| EMG    | 700 Hz        |
| EDA    | 4 Hz          |
| Temp   | 4 Hz          |
| ACC    | 32 Hz         |

Future multimodal analysis may require resampling.

---

# Future Work

Planned additions:

* Random Forest comparison
* Stress state classification
* Sliding windows
* Participant normalization
* Multithreading
* Additional physiological features
* Visualization and plots

---

# Authors

Group 3:
- Chawan Sreelekha
- Indupriya Uppaluri
- Virginia Vaughan

Developed for graduate research in physiological stress forecasting and human performance analytics.


```
