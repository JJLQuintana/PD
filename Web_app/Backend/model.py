import pandas as pd
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

#data load and preprocessing

df = pd.read_csv("test_pcap_625.csv", delimiter='\t')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)  

input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
input_tensor = input_tensor.unsqueeze(0)

# define the model (Vanilla Autoencoder)
class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VanillaAutoencoder, self).__init__()
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

#load model  
input_size = input_tensor.shape[2]  # number of features
model = LSTMClassifier(input_size=input_size)

model = BEST_LSTM_VANILLAAE_MODEL.pth()
PATH = "Web_app\Backend\BEST_LSTM_VANILLAAE_MODEL.pth"
model.load_state_dict(torch.load(PATH))
model.eval()


with torch.no_grad():
    output = model(input_tensor)
    prediction = (output > 0.5).int()  # 1 = DoS, 0 = Benign
    print("Prediction:", "DoS" if prediction.item() == 1 else "Benign")

