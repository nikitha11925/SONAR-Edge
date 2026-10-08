"""
Edge C/C++ Model Exporter for ESP32-S3 Microcontroller
------------------------------------------------------
Extracts trained PyTorch Autoencoder weights and calibration metadata,
generating a self-contained, zero-dependency C/C++ header file (`edge_autoencoder_model.h`)
ready for direct flashing onto an ESP32-S3 firmware project.
"""

import os
import argparse
import numpy as np
import torch
from train_edge_autoencoder import EdgeAutoencoder

def format_c_array_1d(arr, name, dtype="float"):
    lines = [f"static const {dtype} {name}[{len(arr)}] = {{"]
    row_strs = [f"{val:.8f}f" for val in arr]
    for i in range(0, len(row_strs), 8):
        lines.append("    " + ", ".join(row_strs[i:i+8]) + ("," if i+8 < len(row_strs) else ""))
    lines.append("};\n")
    return "\n".join(lines)

def format_c_array_2d(arr, name, dtype="float"):
    rows, cols = arr.shape
    lines = [f"static const {dtype} {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        row_vals = [f"{val:.8f}f" for val in arr[r]]
        lines.append("    { " + ", ".join(row_vals) + " }" + ("," if r < rows - 1 else ""))
    lines.append("};\n")
    return "\n".join(lines)

def export_c_header(model_path, metadata_dir, output_header_path):
    # Load PyTorch Model
    model = EdgeAutoencoder(input_dim=67, bottleneck_dim=16)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load calibration & metadata
    mu_cal = np.load(os.path.join(metadata_dir, 'calibration_mean.npy'))
    sigma_cal = np.load(os.path.join(metadata_dir, 'calibration_std.npy'))
    meta_dict = np.load(os.path.join('model_output', 'model_metadata.npy'), allow_pickle=True).item()
    tau = meta_dict['tau']
    
    # Extract weights and biases
    w_enc1 = model.encoder[0].weight.detach().numpy()  # [32, 67]
    b_enc1 = model.encoder[0].bias.detach().numpy()    # [32]
    w_enc2 = model.encoder[2].weight.detach().numpy()  # [16, 32]
    b_enc2 = model.encoder[2].bias.detach().numpy()    # [16]
    
    w_dec1 = model.decoder[0].weight.detach().numpy()  # [32, 16]
    b_dec1 = model.decoder[0].bias.detach().numpy()    # [32]
    w_dec2 = model.decoder[2].weight.detach().numpy()  # [67, 32]
    b_dec2 = model.decoder[2].bias.detach().numpy()    # [67]
    
    total_params = (w_enc1.size + b_enc1.size + w_enc2.size + b_enc2.size +
                    w_dec1.size + b_dec1.size + w_dec2.size + b_dec2.size)
    
    header_content = f"""/*
 * ==============================================================================
 * Auto-Generated Edge TinyML Autoencoder Model for ESP32-S3
 * Multi-Modal Machinery Anomaly Detection (64 Log-Mel + 3 Vibration RMS)
 * Total Learnable Parameters: {total_params} (~{total_params * 4 / 1024:.2f} KB FP32)
 * Decision Threshold Tau (3-Sigma): {tau:.6f}
 * ==============================================================================
 */

#ifndef EDGE_AUTOENCODER_MODEL_H_
#define EDGE_AUTOENCODER_MODEL_H_

#ifdef __cplusplus
extern "C" {{
#endif

#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#define INPUT_FEATURE_DIM       67
#define ENCODER_HIDDEN_DIM      32
#define BOTTLENECK_DIM          16
#define DECODER_HIDDEN_DIM      32
#define ANOMALY_THRESHOLD_TAU   {tau:.6f}f

// --- Normalization Calibration Parameters ---
{format_c_array_1d(mu_cal, "CALIBRATION_MEAN")}
{format_c_array_1d(sigma_cal, "CALIBRATION_STD")}

// --- Encoder Weights & Biases ---
{format_c_array_2d(w_enc1, "WEIGHTS_ENC_L1")}
{format_c_array_1d(b_enc1, "BIAS_ENC_L1")}
{format_c_array_2d(w_enc2, "WEIGHTS_ENC_L2")}
{format_c_array_1d(b_enc2, "BIAS_ENC_L2")}

// --- Decoder Weights & Biases ---
{format_c_array_2d(w_dec1, "WEIGHTS_DEC_L1")}
{format_c_array_1d(b_dec1, "BIAS_DEC_L1")}
{format_c_array_2d(w_dec2, "WEIGHTS_DEC_L2")}
{format_c_array_1d(b_dec2, "BIAS_DEC_L2")}

/**
 * @brief Run inference on a 67-element fused multi-modal vector.
 * 
 * @param raw_input    Array of 67 raw features (64 Log-Mel bands + 3 Vibration RMS)
 * @param is_anomaly   Pointer to integer set to 1 if anomaly detected (MSE > tau), else 0
 * @return float       The computed Mean Squared Error (MSE) reconstruction loss
 */
static inline float run_edge_autoencoder_inference(const float raw_input[INPUT_FEATURE_DIM], int* is_anomaly) {{
    float z_input[INPUT_FEATURE_DIM];
    float h1[ENCODER_HIDDEN_DIM];
    float h2[BOTTLENECK_DIM];
    float h3[DECODER_HIDDEN_DIM];
    float z_hat[INPUT_FEATURE_DIM];
    
    // 1. On-Device Z-Score Normalization
    for (int i = 0; i < INPUT_FEATURE_DIM; i++) {{
        z_input[i] = (raw_input[i] - CALIBRATION_MEAN[i]) / (CALIBRATION_STD[i] + 1e-6f);
    }}
    
    // 2. Encoder Layer 1 (67 -> 32) + ReLU
    for (int i = 0; i < ENCODER_HIDDEN_DIM; i++) {{
        float sum = BIAS_ENC_L1[i];
        for (int j = 0; j < INPUT_FEATURE_DIM; j++) {{
            sum += WEIGHTS_ENC_L1[i][j] * z_input[j];
        }}
        h1[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }}
    
    // 3. Encoder Layer 2 / Bottleneck (32 -> 16) + ReLU
    for (int i = 0; i < BOTTLENECK_DIM; i++) {{
        float sum = BIAS_ENC_L2[i];
        for (int j = 0; j < ENCODER_HIDDEN_DIM; j++) {{
            sum += WEIGHTS_ENC_L2[i][j] * h1[j];
        }}
        h2[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }}
    
    // 4. Decoder Layer 1 (16 -> 32) + ReLU
    for (int i = 0; i < DECODER_HIDDEN_DIM; i++) {{
        float sum = BIAS_DEC_L1[i];
        for (int j = 0; j < BOTTLENECK_DIM; j++) {{
            sum += WEIGHTS_DEC_L1[i][j] * h2[j];
        }}
        h3[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
    }}
    
    // 5. Decoder Layer 2 / Output (32 -> 67) Linear
    float mse_loss = 0.0f;
    for (int i = 0; i < INPUT_FEATURE_DIM; i++) {{
        float sum = BIAS_DEC_L2[i];
        for (int j = 0; j < DECODER_HIDDEN_DIM; j++) {{
            sum += WEIGHTS_DEC_L2[i][j] * h3[j];
        }}
        z_hat[i] = sum;
        
        float diff = z_input[i] - z_hat[i];
        mse_loss += diff * diff;
    }}
    
    mse_loss /= (float)INPUT_FEATURE_DIM;
    
    // 6. Threshold Classification
    if (is_anomaly != 0) {{
        *is_anomaly = (mse_loss > ANOMALY_THRESHOLD_TAU) ? 1 : 0;
    }}
    
    return mse_loss;
}}

#ifdef __cplusplus
}}
#endif

#endif // EDGE_AUTOENCODER_MODEL_H_
"""
    
    with open(output_header_path, 'w') as f:
        f.write(header_content)
        
    print(f"[SUCCESS] Exported standalone C/C++ header: {output_header_path}")
    print(f"  Model Parameters: {total_params} floats (~{total_params * 4 / 1024:.2f} KB)")
    print(f"  Decision Threshold (tau): {tau:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export Edge Autoencoder to C/C++ Header.")
    parser.add_argument('--model_path', type=str, default='model_output/best_edge_autoencoder.pth')
    parser.add_argument('--metadata_dir', type=str, default='dataset/multimodal_fused')
    parser.add_argument('--output_header', type=str, default='edge_autoencoder_model.h')
    args = parser.parse_args()
    
    export_c_header(args.model_path, args.metadata_dir, args.output_header)
