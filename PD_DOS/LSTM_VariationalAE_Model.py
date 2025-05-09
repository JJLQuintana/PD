# %% [markdown]
# ### Imports

# %%
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

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
df.columns

# %%
unique_values = df[' Label'].unique()
print(unique_values)

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
X = df.drop(columns=['Label'])
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

# %% [markdown]
# ### Creating dataset and loaders

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

# %% [markdown]
# ### Model (LSTM + Vanilla Autoencoder)

# %%
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()

        # More powerful encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.SiLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # More powerful decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp mu and logvar to avoid extreme values
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

# %%
def vae_loss(x, x_reconstructed, mu, logvar):
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='mean')

    # KL divergence between the learned latent distribution and a unit Gaussian
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_divergence

# %%
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# %%
# model initialization
input_dim = X_train.shape[1]
latent_dim = 32

autoencoder = VAE(input_dim, latent_dim).to(device)
lstm = LSTMClassifier(input_size=latent_dim).to(device)

ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCE with manual sigmoid

# %%
# start training time
train_start = time.time()

# hyperparameters
num_epochs = 50
patience = 5
save_path = 'BEST_LSTM_VariationalAE_MODEL.pth'

# early stopping variables
best_acc = 0
epochs_no_improve = 0

# metric storage
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []
train_f1s, test_f1s = [], []

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
        y_batch = y_batch.to(device).unsqueeze(1).float()

        ae_optimizer.zero_grad()
        lstm_optimizer.zero_grad()

        # Forward pass through autoencoder
        x_reconstructed, mu, logvar, z = autoencoder(X_batch)
        z_seq = z.unsqueeze(1)

        # Forward pass through LSTM
        y_pred = lstm(z_seq)

        # VAE loss
        reconstruction_kl_loss = vae_loss(X_batch, x_reconstructed, mu, logvar)

        # Apply BCE loss with logits (no need to apply sigmoid manually)
        loss = reconstruction_kl_loss + criterion(y_pred, y_batch)
        loss.backward()

        # Step for both optimizers
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
            y_batch = y_batch.to(device).unsqueeze(1).float()

            # Forward pass through autoencoder
            x_reconstructed, mu, logvar, z = autoencoder(X_batch)
            z_seq = z.unsqueeze(1)

            # Forward pass through LSTM
            y_pred = lstm(z_seq)

            # VAE loss
            reconstruction_kl_loss = vae_loss(X_batch, x_reconstructed, mu, logvar)

            # Apply BCE loss with logits (no need to apply sigmoid manually)
            test_loss += reconstruction_kl_loss + criterion(y_pred, y_batch)

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

    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)

# end training time
train_end = time.time()
training_time = train_end - train_start

# estimate inference time
inference_start = time.time()
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        _, _, _, z = autoencoder(X_batch)  # FIXED
        z_seq = z.unsqueeze(1)
        _ = lstm(z_seq)
inference_end = time.time()
inference_time = inference_end - inference_start

# %%
# model file size
model_size_kb = os.path.getsize(save_path) / 1024 if os.path.exists(save_path) else 0.0
misclassification_rate = 100 - best_acc

# maintainability index
with open('LSTM_VariationalAE_Model.py', 'r') as f:
    code = f.read()
blocks = cc_visit(code)
mi_score = mi_visit(code, blocks)

print("\n=== Summary ===")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Storage Consumption: {model_size_kb:.2f} KB")
print(f"Misclassification Rate: {misclassification_rate:.2f}")
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Maintainability Index: {mi_score:.2f}")

# %% [markdown]
# ### Plots and Results

# %%
# confusion matrix and roc curve
autoencoder.eval()
lstm.eval()
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        x_reconstructed, mu, logvar, z = autoencoder(X_batch)
        z_seq = z.unsqueeze(1)
        y_pred = lstm(z_seq)

        probs = y_pred.cpu().numpy()
        preds = (y_pred > 0.5).float().cpu().numpy()
        labels = y_batch.cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# %%


# %%



