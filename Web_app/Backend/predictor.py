from flask import Flask, jsonify, render_template
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)  # ← Moved up here before any route usage

@app.route("/")
def home():
    return render_template("index.html")

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
# Load Model Once
# =====================
MODEL_PATH = r"C:\Users\chris\Downloads\sampol\PD\Web_app\Backend\BEST_LSTM_VANILLAAE_MODEL.pth"
CSV_PATH = r"C:\Users\chris\Downloads\sampol\PD\Web_app\Backend\Test.pcap_ISCX.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Inference Route
# =====================
@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    attack_labels = ('DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'Heartbleed')
    df['Label'] = df['Label'].apply(lambda x: 1 if any(attack in x for attack in attack_labels) else 0)

    df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

    X = df.drop(columns=['Label'])
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    model = load_model(input_dim=X.shape[1])

    with torch.no_grad():
        _, output = model(input_tensor)
        output = torch.sigmoid(output)  # In case sigmoid is missing from model
        output = output.squeeze()

        print("Raw model outputs (first 10):", output[:10].cpu().numpy())  # Log confidence scores
        print("DoS samples:", (y == 1).sum(), "| Benign samples:", (y == 0).sum())

        THRESHOLD = 0.5
        predictions = (output > THRESHOLD).int().cpu().numpy().flatten().tolist()

        dos_count = sum(1 for p in predictions if p == 1)
        benign_count = len(predictions) - dos_count
        print(f"Predicted DoS: {dos_count} | Benign: {benign_count}")



        result = [
        {
            "sample": i,
            "label": "DoS" if pred == 1 else "Benign",
            "confidence": float(output[i].cpu())
        }
        for i, pred in enumerate(predictions)
    ]


    return jsonify(result)

# =====================
# Run App
# =====================
if __name__ == "__main__":
    app.run(debug=True)
