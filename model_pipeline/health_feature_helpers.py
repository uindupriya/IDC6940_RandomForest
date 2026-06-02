import neurokit2 as nk

def compute_hr(signal):
    signals, _ = nk.ecg_process(
        signal,
        sampling_rate=700
    )

    return signals["ECG_Rate"]

FEATURE_MAP = {
    "ECG": ["HR", "HRV"],
    "EDA": ["SCR", "SCL"],
    "Resp": ["RespRate"]
}

FEATURE_HELPERS = {
    "HR": compute_hr,
    # "HRV": compute_hrv,
    # "SCR": compute_scr,
    # "SCL": compute_scl,
    # "RespRate": compute_resp_rate
}

