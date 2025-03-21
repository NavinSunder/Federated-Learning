import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

# Load dataset
file_path = "C:\\Amrita cse\\8th Semester\\wustl-ehms-2020_with_attacks_categories (1).csv"
df = pd.read_csv(file_path)

network_features = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "SIntPkt", "DIntPkt",
    "Dur", "TotPkts", "TotBytes", "Loss", "pLoss", "pSrcLoss", "pDstLoss", "Rate", "Label"
]
df_filtered = df[network_features].dropna()

# Split dataset into multiple clients
num_clients = 3  
client_partitions = np.array_split(df_filtered, num_clients)

# Get client ID from command-line arguments
if len(sys.argv) != 2:
    print("Usage: python client.py <client_id>")
    sys.exit(1)

client_id = int(sys.argv[1])
client_data = client_partitions[client_id]

# Convert the data to NumPy arrays
X = client_data.drop(columns=["Label"]).values
y = client_data["Label"].values

# ✅ Convert labels to strictly 0 or 1
y = (y > 0).astype(np.float32)

# ✅ Feature Scaling (Standardization)
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1)

# Create DataLoaders for mini-batch training
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# Define the Model
class linSVM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super(linSVM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x).squeeze()

# Initialize model
input_dim = X_train.shape[1]
model = linSVM(input_dim=input_dim)

# ✅ Use He Initialization for better convergence
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He Initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01)

model.apply(init_weights)

# ✅ Change Loss Function for Better Stability
class_weights = torch.tensor([2.0])  # Adjusted for imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# ✅ Optimized Learning Rate Strategy
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Convergence Threshold
LOSS_THRESHOLD = 0.001  
previous_loss = float("inf")

# Flower Client Class
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        global previous_loss
        self.set_parameters(parameters)
        model.train()
        total_loss = 0.0

        for epoch in range(50):  
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()

                # ✅ Clamp to prevent numerical issues
                outputs = torch.clamp(outputs, min=-10, max=10)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        # ✅ Stop early if improvement is minimal
        if abs(previous_loss - avg_loss) < LOSS_THRESHOLD:
            print(f"Converged at loss {avg_loss}. Stopping early.")
            return self.get_parameters(config), len(X_train), {"early_stop": True}

        previous_loss = avg_loss
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze()

                
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(y_batch.numpy())
                total_loss += criterion(outputs, y_batch).item()

        accuracy = accuracy_score(np.array(all_labels).flatten(), np.array(all_predictions).flatten())
        avg_loss = total_loss / len(test_loader)

        
        print("Unique Predictions:", np.unique(all_predictions, return_counts=True))
        print("Unique True Labels:", np.unique(all_labels, return_counts=True))

        
        class_report_str = classification_report(
            np.array(all_labels).flatten(),
            np.array(all_predictions).flatten(),
            zero_division=1
        )

        print("\n--- Classification Report ---\n", class_report_str)  

        return float(avg_loss), len(X_test), {
        "accuracy": accuracy,
        "loss": avg_loss
    }

# Start Flower Client
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
