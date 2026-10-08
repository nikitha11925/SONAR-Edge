"""
CWRU Bearing Vibration Dataset Cleaning & Feature Extraction Pipeline
--------------------------------------------------------------------
This script implements signal conditioning, DC offset removal, Butterworth
bandpass filtering, vector-accelerated Hampel spike removal, sliding-window
framing, and feature extraction (RMS, Peak-to-Peak, Kurtosis, Crest Factor)
for edge anomaly detection.
"""

import os
import glob
import argparse
import numpy as np
import scipy.io as sio
from scipy import signal, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_mat_file(file_path):
    """
    Load a CWRU .mat file and extract available channels (DE, FE, BA) and RPM.
    """
    mat_data = sio.loadmat(file_path)
    extracted = {
        'filename': os.path.basename(file_path),
        'DE': None,
        'FE': None,
        'BA': None,
        'RPM': 1797
    }
    
    for key, value in mat_data.items():
        if key.startswith('__'):
            continue
        key_upper = key.upper()
        if 'DE_TIME' in key_upper:
            extracted['DE'] = value.flatten().astype(np.float64)
        elif 'FE_TIME' in key_upper:
            extracted['FE'] = value.flatten().astype(np.float64)
        elif 'BA_TIME' in key_upper:
            extracted['BA'] = value.flatten().astype(np.float64)
        elif 'RPM' in key_upper:
            try:
                extracted['RPM'] = int(value.flatten()[0])
            except Exception:
                pass

    return extracted

def remove_dc_offset(sig):
    """Remove DC bias / mean centering: y[t] = x[t] - mean(x)"""
    return sig - np.mean(sig)

def apply_butterworth_bandpass(sig, fs=12000, lowcut=10.0, highcut=5000.0, order=4):
    """
    Zero-phase 4th order Butterworth bandpass filter.
    Suppresses <10 Hz static drift/sway and >5000 Hz anti-aliasing high frequency noise.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if high >= 1.0:
        high = 0.99
    b, a = signal.butter(order, [low, high], btype='bandpass')
    clean_sig = signal.filtfilt(b, a, sig)
    return clean_sig

def hampel_filter(sig, window_size=21, n_sigmas=3.0):
    """
    Vectorized C-accelerated Hampel filter using scipy.ndimage.median_filter.
    """
    k = 1.4826  # Scale factor for Gaussian distribution
    med = ndimage.median_filter(sig, size=window_size)
    mad = k * ndimage.median_filter(np.abs(sig - med), size=window_size)
    outliers = np.abs(sig - med) > (n_sigmas * (mad + 1e-8))
    cleaned_sig = np.where(outliers, med, sig)
    return cleaned_sig

def clean_signal_pipeline(sig, fs=12000, lowcut=10.0, highcut=5000.0):
    """Complete signal cleaning pipeline for 1D vibration array."""
    if sig is None or len(sig) == 0:
        return None
    
    # 1. Initial DC Offset removal
    s1 = remove_dc_offset(sig)
    # 2. Hampel outlier spike removal
    s2 = hampel_filter(s1)
    # 3. Butterworth bandpass filtering
    s3 = apply_butterworth_bandpass(s2, fs=fs, lowcut=lowcut, highcut=highcut)
    # 4. Final DC zeroing after filter boundary effects
    s4 = remove_dc_offset(s3)
    
    return s4

def frame_signal(sig, frame_size=384, hop_size=192, apply_window=True):
    """
    Slice continuous signal into overlapping frames.
    For fs=12,000 Hz: 32 ms = 384 samples; 50% overlap = 192 samples.
    """
    n_samples = len(sig)
    if n_samples < frame_size:
        return np.array([])
    
    n_frames = 1 + (n_samples - frame_size) // hop_size
    frames = np.zeros((n_frames, frame_size), dtype=np.float32)
    
    window = np.hanning(frame_size) if apply_window else np.ones(frame_size)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i] = sig[start:end] * window
        
    return frames

def extract_frame_features(frames):
    """
    Extract statistical time-domain features per frame:
    - RMS (Root Mean Square acceleration)
    - Peak-to-Peak amplitude
    - Kurtosis
    - Crest Factor
    """
    if len(frames) == 0:
        return None
    
    # RMS: sqrt(mean(x^2))
    rms = np.sqrt(np.mean(frames**2, axis=1))
    
    # Peak-to-Peak
    p2p = np.ptp(frames, axis=1)
    
    # Kurtosis: E[(x - mu)^4] / sigma^4
    mu = np.mean(frames, axis=1, keepdims=True)
    std = np.std(frames, axis=1, keepdims=True) + 1e-8
    kurt = np.mean(((frames - mu) / std)**4, axis=1) - 3.0  # Excess kurtosis
    
    # Crest Factor: Peak / RMS
    peak = np.max(np.abs(frames), axis=1)
    crest_factor = peak / (rms + 1e-8)
    
    return {
        'rms': rms,
        'p2p': p2p,
        'kurtosis': kurt,
        'crest_factor': crest_factor
    }

def process_cwru_dataset(input_dir, output_dir, fs=12000, frame_size=384, hop_size=192):
    """
    Main processing loop across all CWRU .mat files.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    mat_files = sorted(glob.glob(os.path.join(input_dir, '*.mat')))
    print(f"[INFO] Found {len(mat_files)} .mat files in {input_dir}", flush=True)
    
    dataset_summary = {}
    processed_records = []
    
    for mat_path in mat_files:
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        parsed = parse_mat_file(mat_path)
        print(f"\n[PROCESSING] {base_name}.mat (RPM: {parsed['RPM']})", flush=True)
        
        # Determine condition label
        if '97' in base_name:
            condition = 'Healthy_Baseline'
            label = 0
        elif '105' in base_name:
            condition = 'Inner_Race_Fault_7mil'
            label = 1
        elif '118' in base_name:
            condition = 'Ball_Fault_7mil'
            label = 2
        elif '130' in base_name:
            condition = 'Outer_Race_Fault_7mil'
            label = 3
        else:
            condition = 'Unknown_Fault'
            label = -1
            
        # Clean channels
        cleaned_channels = {}
        for ch in ['DE', 'FE', 'BA']:
            raw_sig = parsed[ch]
            if raw_sig is not None:
                cleaned_sig = clean_signal_pipeline(raw_sig, fs=fs)
                cleaned_channels[ch] = cleaned_sig
                print(f"  Channel {ch}: Raw length = {len(raw_sig)}, Cleaned length = {len(cleaned_sig)}", flush=True)
            else:
                # If BA is missing (like in 97.mat), simulate baseline floor noise
                if ch == 'BA':
                    ref_len = len(cleaned_channels['DE']) if 'DE' in cleaned_channels else 120000
                    cleaned_channels['BA'] = np.random.normal(0, 0.005, ref_len).astype(np.float64)
                    print(f"  Channel BA: Synthesized baseline floor noise (length={ref_len})", flush=True)
                else:
                    cleaned_channels[ch] = None
        
        # Align channel lengths
        min_len = min(len(cleaned_channels[ch]) for ch in ['DE', 'FE', 'BA'] if cleaned_channels[ch] is not None)
        for ch in ['DE', 'FE', 'BA']:
            if cleaned_channels[ch] is not None:
                cleaned_channels[ch] = cleaned_channels[ch][:min_len]
                
        # Generate Diagnostic Plot (Raw vs Cleaned for DE channel)
        if parsed['DE'] is not None:
            plot_path = os.path.join(plots_dir, f"{base_name}_de_cleaning_plot.png")
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            
            # Time domain
            t_raw = np.arange(min(3000, len(parsed['DE']))) / fs
            axs[0].plot(t_raw, parsed['DE'][:len(t_raw)], color='#e74c3c', label='Raw Signal', alpha=0.8)
            axs[0].plot(t_raw, cleaned_channels['DE'][:len(t_raw)], color='#2ecc71', label='Cleaned & Filtered', linewidth=1.2)
            axs[0].set_title(f"{base_name} ({condition}) - Drive End Time Domain Signal")
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Acceleration (g)")
            axs[0].legend()
            axs[0].grid(True, linestyle='--', alpha=0.5)
            
            # Frequency domain (PSD)
            f_raw, psd_raw = signal.welch(parsed['DE'], fs=fs, nperseg=1024)
            f_clean, psd_clean = signal.welch(cleaned_channels['DE'], fs=fs, nperseg=1024)
            axs[1].semilogy(f_raw, psd_raw, color='#e74c3c', label='Raw PSD', alpha=0.7)
            axs[1].semilogy(f_clean, psd_clean, color='#2ecc71', label='Cleaned PSD', linewidth=1.2)
            axs[1].axvline(10, color='blue', linestyle=':', label='10 Hz Low Cutoff')
            axs[1].axvline(5000, color='purple', linestyle=':', label='5000 Hz High Cutoff')
            axs[1].set_title("Power Spectral Density (PSD)")
            axs[1].set_xlabel("Frequency (Hz)")
            axs[1].set_ylabel("Power / Frequency ($g^2$/Hz)")
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  Saved diagnostic plot: {plot_path}", flush=True)
            
        # Frame each channel
        frames_de = frame_signal(cleaned_channels['DE'], frame_size=frame_size, hop_size=hop_size)
        frames_fe = frame_signal(cleaned_channels['FE'], frame_size=frame_size, hop_size=hop_size)
        frames_ba = frame_signal(cleaned_channels['BA'], frame_size=frame_size, hop_size=hop_size)
        
        n_frames = len(frames_de)
        print(f"  Generated {n_frames} frames (Frame size: {frame_size}, Hop: {hop_size})", flush=True)
        
        # Extract features per frame
        feat_de = extract_frame_features(frames_de)
        feat_fe = extract_frame_features(frames_fe)
        feat_ba = extract_frame_features(frames_ba)
        
        # 3-Axis / 3-Channel RMS feature matrix [n_frames, 3] for 67-feature fusion
        vibration_rms_3ch = np.stack([feat_de['rms'], feat_fe['rms'], feat_ba['rms']], axis=1)
        
        # Save individual file features
        record = {
            'base_name': base_name,
            'condition': condition,
            'label': label,
            'n_frames': n_frames,
            'rms_3ch': vibration_rms_3ch,
            'feat_de': feat_de,
            'feat_fe': feat_fe,
            'feat_ba': feat_ba
        }
        processed_records.append(record)
        
        dataset_summary[base_name] = {
            'condition': condition,
            'label': label,
            'frames': n_frames,
            'mean_rms_de': float(np.mean(feat_de['rms'])),
            'mean_rms_fe': float(np.mean(feat_fe['rms'])),
            'mean_rms_ba': float(np.mean(feat_ba['rms'])),
            'mean_kurtosis_de': float(np.mean(feat_de['kurtosis']))
        }

    # Partition dataset into Healthy Train/Val and Fault Test sets
    healthy_records = [r for r in processed_records if r['label'] == 0]
    fault_records = [r for r in processed_records if r['label'] > 0]
    
    # Healthy dataset splitting
    all_healthy_rms = np.concatenate([r['rms_3ch'] for r in healthy_records], axis=0)
    n_healthy = len(all_healthy_rms)
    n_train = int(0.8 * n_healthy)
    
    # Shuffle healthy frames
    np.random.seed(42)
    indices = np.random.permutation(n_healthy)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    train_healthy_vibration = all_healthy_rms[train_idx]
    val_healthy_vibration = all_healthy_rms[val_idx]
    
    # Combine fault test sets
    test_fault_vibration = np.concatenate([r['rms_3ch'] for r in fault_records], axis=0)
    test_fault_labels = np.concatenate([np.full(r['n_frames'], r['label']) for r in fault_records], axis=0)
    
    # Save standard numpy artifacts
    np.save(os.path.join(output_dir, 'vibration_train_healthy.npy'), train_healthy_vibration)
    np.save(os.path.join(output_dir, 'vibration_val_healthy.npy'), val_healthy_vibration)
    np.save(os.path.join(output_dir, 'vibration_test_faults.npy'), test_fault_vibration)
    np.save(os.path.join(output_dir, 'vibration_test_labels.npy'), test_fault_labels)
    
    # Generate Feature Distribution Comparison Plot
    plot_dist_path = os.path.join(plots_dir, "cwru_vibration_feature_distributions.png")
    plt.figure(figsize=(10, 5))
    plt.hist(train_healthy_vibration[:, 0], bins=40, alpha=0.6, label='Healthy Baseline (97)', color='#2ecc71', density=True)
    
    for r in fault_records:
        plt.hist(r['rms_3ch'][:, 0], bins=40, alpha=0.5, label=f"{r['base_name']} ({r['condition']})", density=True)
        
    plt.title("Drive End (DE) Vibration RMS Energy Distribution across Machine States")
    plt.xlabel("RMS Acceleration (g)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_dist_path, dpi=150)
    plt.close()
    print(f"\n[INFO] Saved summary distribution plot: {plot_dist_path}", flush=True)

    print("\n=======================================================", flush=True)
    print(" CWRU DATASET CLEANING & FEATURE EXTRACTION COMPLETE   ", flush=True)
    print("=======================================================", flush=True)
    print(f"Healthy Train Set (80%): {train_healthy_vibration.shape} frames (Shape: [N, 3])", flush=True)
    print(f"Healthy Val Set   (20%): {val_healthy_vibration.shape} frames (Shape: [N, 3])", flush=True)
    print(f"Fault Test Set         : {test_fault_vibration.shape} frames (Shape: [N, 3])", flush=True)
    print("=======================================================\n", flush=True)
    
    return dataset_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean and process CWRU vibration dataset.")
    parser.add_argument('--input_dir', type=str, default='dataset/cwru', help="Directory containing .mat files")
    parser.add_argument('--output_dir', type=str, default='dataset/cwru/processed', help="Output directory for processed files")
    parser.add_argument('--fs', type=int, default=12000, help="Sampling frequency (Hz)")
    parser.add_argument('--frame_size', type=int, default=384, help="Window frame size (samples)")
    parser.add_argument('--hop_size', type=int, default=192, help="Hop size (samples)")
    
    args = parser.parse_args()
    process_cwru_dataset(args.input_dir, args.output_dir, fs=args.fs, frame_size=args.frame_size, hop_size=args.hop_size)
