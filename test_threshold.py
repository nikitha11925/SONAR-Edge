import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. Simulate 60 healthy calibration clips (right-skewed reconstruction errors)
# Most scores are low (around 0.2 - 0.4), with a few natural operational spikes
healthy_scores = np.random.gamma(shape=2.5, scale=0.15, size=60)

# Simulate test data: 100 healthy test clips + 50 real faulty clips
healthy_test = np.random.gamma(shape=2.5, scale=0.15, size=100)
faulty_test = np.random.normal(loc=1.2, scale=0.25, size=50)

# --- METHOD A: Baseline Gaussian Rule (Mean + 3 * Std) ---
mean_val = np.mean(healthy_scores)
std_val = np.std(healthy_scores)
threshold_old = mean_val + 3 * std_val

# --- METHOD B: Robust MAD Rule (Median + 3 * 1.4826 * MAD) ---
median_val = np.median(healthy_scores)
mad_val = np.median(np.abs(healthy_scores - median_val))
# 1.4826 scales MAD to match standard deviation for normal distributions
threshold_mad = median_val + 3 * (1.4826 * mad_val)

# --- Calculate False Alarms & Detection ---
old_fa = np.mean(healthy_test > threshold_old) * 100
mad_fa = np.mean(healthy_test > threshold_mad) * 100
old_recall = np.mean(faulty_test > threshold_old) * 100
mad_recall = np.mean(faulty_test > threshold_mad) * 100

# --- Plot Distribution & Decision Boundaries ---
plt.figure(figsize=(10, 5))

# Plot distributions
plt.hist(healthy_test, bins=25, alpha=0.6, color='seagreen', label='Healthy Test Clips', density=True)
plt.hist(faulty_test, bins=20, alpha=0.5, color='crimson', label='Faulty Test Clips', density=True)

# Plot thresholds
plt.axvline(threshold_old, color='red', linestyle='--', linewidth=2, 
            label=f'Baseline (Mean + 3σ): {threshold_old:.3f} | False Alarms: {old_fa:.1f}%')
plt.axvline(threshold_mad, color='darkblue', linestyle='-', linewidth=2.5, 
            label=f'Proposed (Median + 3×MAD): {threshold_mad:.3f} | False Alarms: {mad_fa:.1f}%')

plt.title('Threshold Optimization: Baseline Gaussian (Mean+3σ) vs. Robust MAD', fontsize=12, fontweight='bold')
plt.xlabel('Reconstruction Error Score', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('threshold_comparison.png', dpi=300)
print(f"Plot saved to threshold_comparison.png")
print(f"Old Threshold: {threshold_old:.3f} | FA: {old_fa:.1f}% | Recall: {old_recall:.1f}%")
print(f"MAD Threshold: {threshold_mad:.3f} | FA: {mad_fa:.1f}% | Recall: {mad_recall:.1f}%")