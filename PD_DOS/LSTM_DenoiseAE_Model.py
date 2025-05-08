# %% [markdown]
# # Design of Deep Learning-based Denial-of-Service Attack Detection for an Intrusion Prevention System with Automated Policy Updating

# %% [markdown]
# ## Model Training LSTM + Denoise Autoencoder

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import time
import os
from radon.metrics import mi_visit
from radon.complexity import cc_visit

# %%
# for checking if cuda is working c:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% [markdown]
# ### Dataset (preprocessing)

# %%
df = pd.read_csv('Dataset\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# %%
df.head()

# %%
unique_values = df[' Label'].unique()
print(unique_values)

# %%
df.columns

# %%
df.columns = df.columns.str.strip() # pang tanggal ng space kasi may space sa column names hehe

# %%
df.columns # yey wala na space sa harap

# %%
df['Label'] = df['Label'].apply(lambda x: 1 if 'DDoS' in x else 0)

# %%
check_Label = df['Label'].unique()
print(check_Label) # 0 = Benign, 1 = DDoS

# %%
df = df.replace([float('inf'), float('-inf')], pd.NA).dropna() # dropping rows with nan and inf values

# %% [markdown]
# ### Standardize and Splitting

# %%
X = df.drop(columns=[   'Label'])
y = df['Label']

# %%
X

# %%
y

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # standardize features 

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# %%
df['Label'].value_counts() # 0 = Benign, 1 = DDoS

# %%
# convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# %%
# create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# %%
# create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, noise_std=0.1):  # Increase latent_dim
        super(DenoisingAutoencoder, self).__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased number of hidden neurons
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        # Add Gaussian noise during training only
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


# LSTM classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


# %%
# model initialization
input_dim = X_train.shape[1]
latent_dim = 32
noise_std = 0.1 

autoencoder = DenoisingAutoencoder(input_dim=input_dim, latent_dim=latent_dim, noise_std=noise_std).to(device)
lstm = LSTMClassifier(input_size=latent_dim).to(device)

ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
criterion = nn.BCELoss()


# %%
# start training time
train_start = time.time()

# hyperparameters
num_epochs = 50
patience = 5
save_path = 'BEST_LSTM_DENOISEAE_MODEL.pth'

# early stopping variables
best_acc = 0
epochs_no_improve = 0

# metric storage
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
train_f1s, test_f1s = [], []

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    autoencoder.train()
    lstm.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        ae_optimizer.zero_grad()
        lstm_optimizer.zero_grad()

        # Add noise for denoising AE
        noisy_input = X_batch + autoencoder.noise_std * torch.randn_like(X_batch)
        noisy_input = torch.clamp(noisy_input, 0., 1.)

        _, z = autoencoder(noisy_input)
        z_seq = z.unsqueeze(1)
        y_pred = lstm(z_seq)

        loss = criterion(y_pred, y_batch)
        loss.backward()
        ae_optimizer.step()
        lstm_optimizer.step()

        total_loss += loss.item()
        predicted = (y_pred > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct / total * 100
    train_f1 = f1_score(all_labels, all_preds)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    train_f1s.append(train_f1)

    # eval on test set
    autoencoder.eval()
    lstm.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            # No noise added during inference
            _, z = autoencoder(X_batch)
            z_seq = z.unsqueeze(1)
            y_pred = lstm(z_seq)

            loss = criterion(y_pred, y_batch)
            test_loss += loss.item()

            predicted = (y_pred > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = correct / total * 100
    test_f1 = f1_score(all_labels, all_preds)
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_acc)
    test_f1s.append(test_f1)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%, F1: {train_f1:.4f}")
    print(f"  Test  -> Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%, F1: {test_f1:.4f}")

    # Early stopping and model checkpoint
    if test_acc > best_acc:
        best_acc = test_acc
        epochs_no_improve = 0
        torch.save({
            'autoencoder_state_dict': autoencoder.state_dict(),
            'lstm_state_dict': lstm.state_dict(),
            'ae_optimizer_state_dict': ae_optimizer.state_dict(),
            'lstm_optimizer_state_dict': lstm_optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }, save_path)
        print(f"Model saved at epoch {epoch+1} with accuracy: {best_acc:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

# end training time
train_end = time.time()
training_time = train_end - train_start

# estimate inference time
inference_start = time.time()
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        _, z = autoencoder(X_batch)
        z_seq = z.unsqueeze(1)
        _ = lstm(z_seq)
inference_end = time.time()
inference_time = inference_end - inference_start

# model file size
model_size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0.0

# maintainability index
with open('LSTM_DenoiseAE_Model.py', 'r') as f:
    code = f.read()
blocks = cc_visit(code)
mi_score = mi_visit(code, blocks)

# final summary output
print("\n=== Summary ===")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Storage Consumption: {model_size_kb:.2f} KB")
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Maintainability Index: {mi_score:.2f}")
print("All tasks completed.")


# %%
autoencoder.eval()
lstm.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        # For Denoising Autoencoder, no noise is added during eval
        _, z = autoencoder(X_batch)
        z_seq = z.unsqueeze(1)
        y_pred = lstm(z_seq)

        # Get probabilities and binary predictions
        probs = y_pred.cpu().numpy()
        preds = (y_pred > 0.5).float().cpu().numpy()
        labels = y_batch.cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

# %%
# Convert lists to numpy arrays
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# %%
# Misclassification Rate
misclassification_rate = (cm[0][1] + cm[1][0]) / cm.sum()
print(f"Misclassification Rate: {misclassification_rate:.4f}")

# %%
# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(16, 5))

# Loss plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='s')
plt.title('Loss over Epochs (Denoising AE + LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='s')
plt.title('Accuracy over Epochs (Denoising AE + LSTM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# F1-score plot
plt.subplot(1, 3, 3)
plt.plot(epochs, train_f1s, label='Train F1-Score', marker='o')
plt.plot(epochs, test_f1s, label='Test F1-Score', marker='s')
plt.title('F1 Score over Epochs (Denoising AE + LSTM)')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# %%
# Evaluate the model
autoencoder.eval()
lstm.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        # Forward pass through denoising autoencoder
        _, z = autoencoder(X_batch)
        z_seq = z.unsqueeze(1)

        # Forward pass through stacked LSTM
        y_pred = lstm(z_seq)

        # Threshold prediction
        predicted = (y_pred > 0.5).float()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Generate and print classification report
print("\n=== Classification Report (Denoising AE + LSTM) ===")
print(classification_report(all_labels, all_preds, target_names=['Benign', 'DDoS']))



