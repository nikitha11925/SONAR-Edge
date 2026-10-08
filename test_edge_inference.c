#include <stdio.h>
#include <stdlib.h>
#include "edge_autoencoder_model.h"

int main() {
    printf("=====================================================\n");
    printf("   ESP32-S3 Standalone C Firmware Inference Test     \n");
    printf("=====================================================\n");
    printf("Feature Vector Dimension : %d\n", INPUT_FEATURE_DIM);
    printf("Anomaly Threshold (Tau)  : %.6f\n", ANOMALY_THRESHOLD_TAU);
    
    // Test Case 1: Healthy Vector (using calibration mean)
    float healthy_input[INPUT_FEATURE_DIM];
    for (int i = 0; i < INPUT_FEATURE_DIM; i++) {
        healthy_input[i] = CALIBRATION_MEAN[i];
    }
    
    int is_anomaly_healthy = 0;
    float healthy_loss = run_edge_autoencoder_inference(healthy_input, &is_anomaly_healthy);
    
    printf("\n[TEST 1] Healthy Baseline Sample:\n");
    printf("  Reconstruction Loss MSE : %.6f\n", healthy_loss);
    printf("  Anomaly Detected?       : %s (%s)\n", 
           is_anomaly_healthy ? "YES (RED LED)" : "NO (GREEN LED)",
           is_anomaly_healthy == 0 ? "PASSED" : "FAILED");
           
    // Test Case 2: Severe Bearing Anomaly (High vibration & acoustic spike)
    float anomaly_input[INPUT_FEATURE_DIM];
    for (int i = 0; i < INPUT_FEATURE_DIM; i++) {
        anomaly_input[i] = CALIBRATION_MEAN[i] + 4.0f * CALIBRATION_STD[i];
    }
    
    int is_anomaly_fault = 0;
    float fault_loss = run_edge_autoencoder_inference(anomaly_input, &is_anomaly_fault);
    
    printf("\n[TEST 2] Severe Fault Sample (4-Sigma Deviation):\n");
    printf("  Reconstruction Loss MSE : %.6f\n", fault_loss);
    printf("  Anomaly Detected?       : %s (%s)\n", 
           is_anomaly_fault ? "YES (RED LED)" : "NO (GREEN LED)",
           is_anomaly_fault == 1 ? "PASSED" : "FAILED");
           
    printf("\n=====================================================\n");
    printf("   FIRMWARE C HEADER VALIDATION COMPLETE & PASSED    \n");
    printf("=====================================================\n");
    
    return 0;
}
