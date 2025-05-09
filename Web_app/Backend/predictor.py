import torch
import torch.nn as nn
from preprocessing import preprocess_input

class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VanillaAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),   # ← Match checkpoint
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),  # ← Match checkpoint
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

def predict_dos():
    # 1. Load and preprocess input
    X_scaled, _ = preprocess_input()
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 2. Load checkpoint
    checkpoint = torch.load("BEST_LSTM_DENOISEAE_MODEL.pth", map_location=torch.device('cpu'))

    # 3. Rebuild models
    input_dim = input_tensor.shape[1]
    ae = VanillaAutoencoder(input_dim=input_dim)
    ae.load_state_dict(checkpoint["autoencoder_state_dict"])
    ae.eval()

    lstm_input_dim = 32  # Must match encoder output
    model = LSTMClassifier(input_size=lstm_input_dim)
    model.load_state_dict(checkpoint["lstm_state_dict"])
    model.eval()

    # 4. Encode input to latent space
    with torch.no_grad():
        _, z = ae(input_tensor)
        z = z.unsqueeze(0)  # LSTM expects batch_size x seq_len x features
        output = model(z)
        prediction = (output > 0.5).int().item()
        return "DoS" if prediction == 1 else "Benign"
