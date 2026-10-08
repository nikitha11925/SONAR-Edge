
#include "edge_autoencoder_model.h"
#define EXPORT __declspec(dllexport)

EXPORT float test_c_inference(const float* raw_input, int* is_anomaly) {
    return run_edge_autoencoder_inference(raw_input, is_anomaly);
}
