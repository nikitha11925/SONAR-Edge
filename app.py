import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import torch
import torch.nn as nn

st.set_page_config(
    page_title="SONAR-Edge | Vibration Diagnostics (Small1DCNN + Hilbert)",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.2rem;
    }
    .status-banner {
        padding: 16px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Exact Architecture from train_cnn.py (<20k params) ---
class Small1DCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=2, padding=31, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@st.cache_resource
def load_cnn_model(weights_path: str, num_classes: int):
    if not os.path.exists(weights_path):
        return None
    model = Small1DCNN(in_channels=1, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_cwru_testset(npz_path: str):
    if not os.path.exists(npz_path):
        return None, None, None
    data = np.load(npz_path, allow_pickle=True)
    if "X_test" in data:
        X_test = data["X_test"]
        y_test = data["y_test"]
        classes = [str(c) for c in data["classes"]]
    else:
        keys = list(data.keys())
        X_test = data[keys[0]]
        y_test = data[keys[1]] if len(keys) > 1 else np.zeros(len(X_test), dtype=int)
        classes = [f"Class_{i}" for i in range(int(np.max(y_test)) + 1)]
    return X_test, y_test, classes


def extract_hilbert_envelope(signal: np.ndarray, fs: float = 48000.0):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    env_ac = envelope - np.mean(envelope)
    n = len(env_ac)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(env_ac)) / n
    return envelope, freqs, fft_mag


st.title("SONAR-Edge: Real-Time 1D CNN & Hilbert Bearing Diagnostics")
st.caption("Live PyTorch Inference (Small1DCNN <20k Params) | Unseen CWRU Test Set | Hilbert Envelope Demodulation")

# --- Load Dataset & Model ---
X_test, y_test, classes = load_cwru_testset("cwru_clean.npz")
num_classes = len(classes) if classes is not None else 10
model = load_cnn_model("best_model.pt", num_classes=num_classes)

# --- Sidebar Controls ---
st.sidebar.header("1. Vibration Input Source")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=48000, step=1000)

source_mode = st.sidebar.radio(
    "Select Input Mode",
    ["CWRU Unseen Test Set (cwru_clean.npz)", "Upload Custom Signal (.csv / .npy)"],
)

signal = None
ground_truth_str = "Unknown (External File)"

if source_mode == "CWRU Unseen Test Set (cwru_clean.npz)" and X_test is not None:
    st.sidebar.success(f"Loaded X_test: {len(X_test)} windows ({num_classes} classes)")

    # Filter by class so you can demo any specific fault in 1 click
    selected_cls = st.sidebar.selectbox("Filter by True Bearing Condition", ["All Classes"] + classes)
    if selected_cls == "All Classes":
        valid_indices = np.arange(len(X_test))
    else:
        cls_id = classes.index(selected_cls)
        valid_indices = np.where(y_test == cls_id)[0]

    pos = st.sidebar.slider("Test Window Index", 0, len(valid_indices) - 1, 0)
    idx = valid_indices[pos]
    signal = X_test[idx].squeeze()
    true_cls_name = classes[int(y_test[idx])]
    ground_truth_str = f"CWRU Test Window #{idx} — True Label: {true_cls_name}"

else:
    uploaded_file = st.sidebar.file_uploader("Upload 2048-Sample Signal (.csv / .npy)", type=["csv", "npy"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".npy"):
            signal = np.load(uploaded_file).squeeze()
        else:
            signal = np.loadtxt(uploaded_file, delimiter=",").squeeze()
        ground_truth_str = f"Uploaded File: {uploaded_file.name}"

if signal is None:
    t = np.arange(2048) / fs
    impacts = (np.sin(2 * np.pi * 188.0 * t) > 0.85).astype(float)
    signal = impacts * np.sin(2 * np.pi * 3000.0 * t) + 0.12 * np.random.randn(len(t))
    ground_truth_str = "Synthetic 188 Hz BPFO Fallback Signal"

# Ensure 2048 window length for Small1DCNN
if len(signal) > 2048:
    sig_for_cnn = signal[:2048]
elif len(signal) < 2048:
    sig_for_cnn = np.pad(signal, (0, 2048 - len(signal)))
else:
    sig_for_cnn = signal

st.sidebar.divider()
st.sidebar.header("2. MAD Calibration Threshold")
mad_cutoff = st.sidebar.number_input("MAD Alert Cutoff (k=3.0)", value=0.726, step=0.01)

# --- Run Real PyTorch 1D CNN Inference ---
if model is not None:
    st.sidebar.success("Loaded best_model.pt (Real Inference Active)")
    x_tensor = torch.from_numpy(sig_for_cnn).float().view(1, 1, 2048)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx] if classes else f"Class {pred_idx}"
    pred_conf = float(probs[pred_idx])

    # Identify Normal/Healthy class index to compute total Fault Probability = 1 - P(Normal)
    normal_indices = [i for i, c in enumerate(classes or []) if "norm" in c.lower() or "health" in c.lower()]
    if normal_indices:
        p_normal = float(probs[normal_indices[0]])
        fault_prob = 1.0 - p_normal
    else:
        fault_prob = 0.0 if pred_idx == 0 else pred_conf
else:
    st.sidebar.warning("best_model.pt not found in E:\\SONAR\\")
    probs = None
    pred_class = "Model Not Loaded"
    pred_conf = 0.0
    fault_prob = 0.0

# --- Run Hilbert Envelope & FFT ---
envelope, freqs, fft_mag = extract_hilbert_envelope(sig_for_cnn, fs=fs)
bpfo_band = (freqs >= 175) & (freqs <= 200)
bpfo_amp = float(np.max(fft_mag[bpfo_band])) if np.any(bpfo_band) else 0.0
rms_val = float(np.sqrt(np.mean(sig_for_cnn ** 2)))

# --- Status Banner ---
if fault_prob >= mad_cutoff:
    status = f"FAULT DETECTED — Predicted: {pred_class} ({pred_conf*100:.1f}%)"
    bg_hex, border_hex = "#381316", "#F85149"
    detail = f"Aggregate fault probability ({fault_prob:.3f}) exceeds MAD cutoff ({mad_cutoff:.3f})."
elif fault_prob >= 0.40:
    status = f"EARLY DRIFT WATCH — Predicted: {pred_class} ({pred_conf*100:.1f}%)"
    bg_hex, border_hex = "#332408", "#D29922"
    detail = f"Moderate anomaly score ({fault_prob:.3f}) approaching MAD cutoff ({mad_cutoff:.3f})."
else:
    status = f"HEALTHY OPERATION — Predicted: {pred_class} ({pred_conf*100:.1f}%)"
    bg_hex, border_hex = "#0F2E1B", "#2EA043"
    detail = f"Aggregate fault probability ({fault_prob:.3f}) is well below MAD cutoff ({mad_cutoff:.3f})."

st.markdown(
    f"""
    <div class="status-banner" style="background-color:{bg_hex}; border:1px solid {border_hex}; color:{border_hex};">
        {status}
        <div style="font-size:14px; font-weight:400; color:#C9D1D9; margin-top:4px;">
            {ground_truth_str} &nbsp;•&nbsp; {detail}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- KPI Row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Class (Small1DCNN)", pred_class, f"Confidence: {pred_conf*100:.1f}%")
c2.metric("Fault Probability (1 - P_normal)", f"{fault_prob:.3f}", f"MAD Cutoff: {mad_cutoff:.3f}")
c3.metric("BPFO (~188 Hz) Peak Mag", f"{bpfo_amp:.4f}", "Hilbert Envelope FFT")
c4.metric("Window RMS Acceleration", f"{rms_val:.4f} g", "2048 samples @ 48 kHz")

st.divider()

# --- Hilbert Time & FFT Plots ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Time-Domain Vibration vs. Hilbert Envelope")
    fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
    fig1.patch.set_facecolor("#0E1117")
    ax1.set_facecolor("#161B22")
    n_plot = 1200
    t_ms = (np.arange(n_plot) / fs) * 1000.0
    ax1.plot(t_ms, sig_for_cnn[:n_plot], color="#58A6FF", alpha=0.55, lw=0.9, label="Raw Vibration Signal")
    ax1.plot(t_ms, envelope[:n_plot], color="#F85149", lw=1.8, label="Hilbert Envelope")
    ax1.set_xlabel("Time (ms)", color="#C9D1D9")
    ax1.set_ylabel("Amplitude", color="#C9D1D9")
    ax1.tick_params(colors="#8B949E")
    ax1.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")
    ax1.grid(True, alpha=0.15, color="#8B949E")
    st.pyplot(fig1)

with col_right:
    st.subheader("2. Demodulated Envelope FFT Spectrum")
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.5))
    fig2.patch.set_facecolor("#0E1117")
    ax2.set_facecolor("#161B22")
    mask = freqs <= 600
    ax2.plot(freqs[mask], fft_mag[mask], color="#3FB950", lw=1.6, label="Envelope FFT Spectrum")
    ax2.axvline(188.0, color="#F85149", ls="--", lw=1.3, label="BPFO (~188 Hz)")
    ax2.axvline(375.0, color="#D29922", ls=":", lw=1.3, label="2x Harmonic (~375 Hz)")
    ax2.set_xlabel("Frequency (Hz)", color="#C9D1D9")
    ax2.set_ylabel("Magnitude", color="#C9D1D9")
    ax2.tick_params(colors="#8B949E")
    ax2.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")
    ax2.grid(True, alpha=0.15, color="#8B949E")
    st.pyplot(fig2)

# --- Live 10-Class Softmax Probability Bar Chart ---
if probs is not None and classes is not None:
    st.subheader("3. Live 10-Class Softmax Output Distribution (best_model.pt)")
    fig3, ax3 = plt.subplots(figsize=(12, 2.4))
    fig3.patch.set_facecolor("#0E1117")
    ax3.set_facecolor("#161B22")
    bar_colors = ["#2EA043" if "norm" in c.lower() else ("#F85149" if i == pred_idx else "#388BFD") for i, c in enumerate(classes)]
    ax3.bar(classes, probs * 100.0, color=bar_colors, width=0.55)
    ax3.set_ylabel("Probability (%)", color="#C9D1D9")
    ax3.set_ylim(0, 105)
    ax3.tick_params(colors="#C9D1D9", axis="x", rotation=20)
    ax3.tick_params(colors="#8B949E", axis="y")
    ax3.grid(True, axis="y", alpha=0.15, color="#8B949E")
    st.pyplot(fig3)