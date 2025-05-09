from flask import Flask, jsonify
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# =====================
# Model Definition
# =====================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        z = z.unsqueeze(1)
        out, _ = self.lstm(z)
        out = self.fc(out[:, -1, :])
        return x_reconstructed, out

# =====================
# Constants and Setup
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "BEST_LSTM_VANILLAAE_MODEL.pth")
CSV_PATH = os.path.join(BASE_DIR, "Test_pcap.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Load Model
# =====================
def load_model(input_dim):
    model = LSTMAutoencoder(input_dim=input_dim, latent_dim=32, hidden_dim=128).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    model_state_dict = checkpoint.get('autoencoder_state_dict', {})
    lstm_state_dict = checkpoint.get('lstm_state_dict', {})

    model.load_state_dict(model_state_dict, strict=False)
    model.lstm.load_state_dict(lstm_state_dict, strict=False)
    model.eval()
    return model

# =====================
# Routes
# =====================

@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    attack_labels = ('DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'Heartbleed')
    df['Label'] = df['Label'].apply(lambda x: 1 if any(attack in x for attack in attack_labels) else 0)

    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

    X = df.drop(columns=['Label'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    model = load_model(input_dim=X.shape[1])

    with torch.no_grad():
        _, output = model(input_tensor)
        predictions = (output > 0.5).int().cpu().numpy().flatten().tolist()

    result = [{"sample": i, "label": "DoS" if pred == 1 else "Benign"} for i, pred in enumerate(predictions)]
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return "Flask app is running. Use /predict to get DoS predictions."

# =====================
# Run
# =====================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
