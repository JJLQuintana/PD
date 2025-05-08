import torch
import torch.nn as nn
import os
from preprocessing import preprocess_input


# Define the combined model (LSTM Autoencoder)
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()

        # Autoencoder part
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

        # LSTM part
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Autoencoder forward pass
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        
        # LSTM forward pass (expecting sequences of latent vectors)
        z = z.unsqueeze(1)  # Adding an extra dimension for LSTM (batch_size, seq_len=1, latent_dim)
        out, _ = self.lstm(z)  # LSTM expects 3D input: (batch_size, seq_len, input_dim)
        out = self.fc(out[:, -1, :])  # Only consider the last output for classification
        
        return x_reconstructed, out

# Model initialization
model = LSTMAutoencoder(input_dim=77, latent_dim=32, hidden_dim=128)

# Check if the directory exists
model_path = r"C:\Users\Jeyo Quintana\Documents\PD\Web_app\Backend\BEST_LSTM_VANILLA_AE_MODEL.pth"
model_dir = os.path.dirname(model_path)

if not os.path.exists(model_dir):
    print(f"Directory does not exist: {model_dir}")
else:
    print(f"Directory exists: {model_dir}")

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Model file does not exist at: {model_path}")
    print(f"Absolute path: {os.path.abspath(model_path)}")
else:
    print(f"Model file found at: {model_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

# The checkpoint seems to have multiple sub-dictionaries
# Let's print the keys in the checkpoint to understand its structure
print("Checkpoint keys:", checkpoint.keys())

# Extracting only the state_dicts we need
model_state_dict = checkpoint.get('autoencoder_state_dict', {})
lstm_state_dict = checkpoint.get('lstm_state_dict', {})

# Now let's load the state_dicts into the model
model.load_state_dict(model_state_dict, strict=False)  # Use strict=False to ignore missing keys
model.lstm.load_state_dict(lstm_state_dict, strict=False)  # Load LSTM part separately

model.eval()  # Set the model to evaluation mode

# Perform inference
with torch.no_grad():
    x_reconstructed, output = model(input_tensor)
    prediction = (output > 0.5).int()  # 1 = DoS, 0 = Benign
    print("Prediction:", "DoS" if prediction.item() == 1 else "Benign")