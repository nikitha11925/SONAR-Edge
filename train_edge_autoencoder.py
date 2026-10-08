"""
Edge Autoencoder Anomaly Detection Engine (PyTorch)
--------------------------------------------------
Trains a compact INT8-friendly Dense Autoencoder (67 -> 32 -> 16 -> 32 -> 67)
on healthy baseline multi-modal data. Computes dynamic calibration threshold
and evaluates ROC-AUC, Precision, Recall, and F1-score across bearing fault test sets.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EdgeAutoencoder(nn.Module):
    def __init__(self, input_dim=67, bottleneck_dim=16):
        super(EdgeAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def compute_reconstruction_mse(model, data_tensor, device):
    model.eval()
    with torch.no_grad():
        x = data_tensor.to(device)
        x_hat = model(x)
        mse_per_sample = torch.mean((x - x_hat)**2, dim=1).cpu().numpy()
    return mse_per_sample

def train_and_evaluate(data_dir, output_dir, epochs=60, batch_size=32, lr=1e-3):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load dataset
    X_train = np.load(os.path.join(data_dir, 'X_train_norm.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val_norm.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_norm.npy'))
    y_test_labels = np.load(os.path.join(data_dir, 'y_test_labels.npy'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using compute device: {device}", flush=True)
    
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    
    # Instantiate Model
    model = EdgeAutoencoder(input_dim=67, bottleneck_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print("\n--- Training Compact Edge Autoencoder (67 -> 32 -> 16 -> 32 -> 67) ---", flush=True)
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_weights_path = os.path.join(output_dir, 'best_edge_autoencoder.pth')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss = running_loss / len(train_tensor)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device))
            epoch_val_loss = criterion(val_recon, val_tensor.to(device)).item()
            
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_weights_path)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train MSE: {epoch_train_loss:.5f} | Val MSE: {epoch_val_loss:.5f}", flush=True)
            
    # Load best model
    model.load_state_dict(torch.load(best_weights_path))
    print(f"\n[INFO] Loaded best model checkpoint from {best_weights_path}", flush=True)
    
    # 1. Dynamic Threshold Calibration on Validation Set
    val_errors = compute_reconstruction_mse(model, val_tensor, device)
    mu_err = np.mean(val_errors)
    sigma_err = np.std(val_errors)
    tau = mu_err + 3.0 * sigma_err  # 3-Sigma Dynamic Threshold
    
    print(f"\n--- On-Device Dynamic Calibration Parameters ---", flush=True)
    print(f"  Validation Baseline Error (Mean) : {mu_err:.5f}", flush=True)
    print(f"  Validation Baseline Error (Std)  : {sigma_err:.5f}", flush=True)
    print(f"  Decision Threshold Tau (3-Sigma) : {tau:.5f}", flush=True)
    
    # 2. Evaluation on Combined Test Set (Healthy Val + Fault Test)
    eval_tensor = torch.cat([val_tensor, test_tensor], dim=0)
    true_binary_labels = np.concatenate([np.zeros(len(val_tensor)), np.ones(len(test_tensor))])
    
    all_errors = compute_reconstruction_mse(model, eval_tensor, device)
    
    # ROC-AUC Metric
    roc_auc = roc_auc_score(true_binary_labels, all_errors)
    fpr, tpr, thresholds = roc_curve(true_binary_labels, all_errors)
    
    # Binary predictions using 3-sigma threshold
    pred_labels = (all_errors > tau).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(true_binary_labels, pred_labels, average='binary')
    cm = confusion_matrix(true_binary_labels, pred_labels)
    
    tn, fp, fn, tp = cm.ravel()
    fpr_rate = fp / (fp + tn + 1e-8)
    tpr_rate = tp / (tp + fn + 1e-8)
    
    print(f"\n=======================================================", flush=True)
    print(f"       EDGE AUTOENCODER ANOMALY DETECTION RESULTS      ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  ROC-AUC Score                   : {roc_auc * 100:.2f}%", flush=True)
    print(f"  True Anomaly Detection Rate (TPR): {tpr_rate * 100:.2f}%", flush=True)
    print(f"  False Positive Rate (FPR)        : {fpr_rate * 100:.2f}%", flush=True)
    print(f"  Precision                       : {precision * 100:.2f}%", flush=True)
    print(f"  Recall                          : {recall * 100:.2f}%", flush=True)
    print(f"  F1-Score                        : {f1 * 100:.2f}%", flush=True)
    print(f"=======================================================\n", flush=True)
    
    # 3. Diagnostic Visualizations
    # Loss Curve Plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Training Loss (MSE)', color='#3498db')
    plt.plot(val_losses, label='Validation Loss (MSE)', color='#e67e22')
    plt.title("Autoencoder Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_loss_curve.png'), dpi=150)
    plt.close()
    
    # ROC Curve Plot
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'Autoencoder (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    # Error Histogram & Threshold Visualization
    plt.figure(figsize=(9, 4.5))
    plt.hist(all_errors[true_binary_labels == 0], bins=30, alpha=0.6, label='Healthy Operation', color='#2ecc71', density=True)
    plt.hist(all_errors[true_binary_labels == 1], bins=40, alpha=0.6, label='Anomalous Faults', color='#e74c3c', density=True)
    plt.axvline(tau, color='black', linestyle='--', linewidth=2, label=f'3-Sigma Threshold (τ = {tau:.3f})')
    plt.title("Reconstruction MSE Anomaly Scoring & Threshold Separation")
    plt.xlabel("Reconstruction MSE Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'anomaly_score_distribution.png'), dpi=150)
    plt.close()
    
    print(f"[SUCCESS] Diagnostic evaluation plots saved to '{plots_dir}'.")
    
    # Save calibration metadata for C-header export
    metadata = {
        'mu_error': float(mu_err),
        'sigma_error': float(sigma_err),
        'tau': float(tau),
        'roc_auc': float(roc_auc),
        'f1': float(f1)
    }
    np.save(os.path.join(output_dir, 'model_metadata.npy'), metadata)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate Edge Autoencoder.")
    parser.add_argument('--data_dir', type=str, default='dataset/multimodal_fused', help="Path to fused multi-modal dataset")
    parser.add_argument('--output_dir', type=str, default='model_output', help="Directory to save trained model & plots")
    parser.add_argument('--epochs', type=int, default=60, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    train_and_evaluate(args.data_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
