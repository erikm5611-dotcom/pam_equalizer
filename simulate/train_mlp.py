# train_mlp.py
import config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load dataset
data = np.load("dataset_pam4_cd_pd.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_val   = data["X_val"]
y_val   = data["y_val"]
X_test  = data["X_test"]
y_test  = data["y_test"]

# first standardize the features to improve performance:
mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-8

X_train = (X_train - mu) / sigma
X_val   = (X_val   - mu) / sigma
X_test  = (X_test  - mu) / sigma

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=256,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=512
)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # input dimension = window size
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


def train_epoch(loader):
    model.train() # training mode
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device) # move to GPU if available

        optimizer.zero_grad() # reset gradients
        logits = model(X) # Compute predicted scores (forward passs)
        loss = criterion(logits, y) # compute the loss
        loss.backward() # Backpropagation (compute gradients for all layers)
        optimizer.step() # Update weights with the gradients we computed

        total_loss += loss.item() * X.size(0) # track the total loss for the epoch

    return total_loss / len(loader.dataset) #  Returns average loss over the entire dataset

def eval_epoch(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

best_val_loss = np.inf
best_epoch = 0

for epoch in range(100): # we train for 100 epochs
    loss = train_epoch(train_loader) # First update weights using training data and compute training loss for this epoch
    val_loss  = eval_epoch(val_loader) # Then evaluate performance on validation data and compute validation loss for this epoch
    print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Val Loss {val_loss:.4f}") # just prints the results for this epoch

    # Save best model basically Whenever validation loss improves, save the model weights
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "best_mlp.pt")

# After training, load the best model and evaluate on test set
print(f"\nBest model from epoch {best_epoch} with Val Loss {best_val_loss:.4f}")

model.load_state_dict(torch.load("best_mlp.pt"))
model.eval()


#compute the training accuracy as well to check for overfitting, to compare it to validation accuracy:
train_loss = eval_epoch(DataLoader(TensorDataset(X_train, y_train), batch_size=512))
print("Train Loss:", train_loss)

with torch.no_grad(): # this runs model on the test set without computing gradients since we are only evaluating
    y_pred = model(X_test.to(device)).cpu().numpy().flatten()

levels = np.array([-3, -1, 1, 3])
rx_hat = levels[np.argmin(np.abs(y_pred[:,None] - levels[None,:]), axis=1)] # map the continuous predictions to the nearest PAM-4 level

# BER 
def pam4_to_bits(sym):
    # must match Gray mapping
    if sym == -3: return [0, 0]
    if sym == -1: return [0, 1]
    if sym ==  1: return [1, 1]
    if sym ==  3: return [1, 0]
    raise ValueError(sym)

y_true = y_test.numpy().flatten()

bit_err = 0
for s_hat, s_true in zip(rx_hat, y_true):
    bh = pam4_to_bits(int(s_hat))
    bt = pam4_to_bits(int(s_true))
    bit_err += (bh[0] != bt[0]) + (bh[1] != bt[1])

ber = bit_err / (2 * len(y_true))
print("Test BER:", ber)

ser = np.mean(rx_hat != y_test.numpy().flatten())
print("Test SER:", ser)


# computational complexity 

# 1. parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters:", count_parameters(model))

# 2. runtime measurement
import time

def measure_inference_time(model, X, n_runs=100):
    model.eval()
    X = X.to(device)

    # Warmup (important for GPU)
    with torch.no_grad():
        for _ in range(10):
            _ = model(X)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(X)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    return (end - start) / n_runs

avg_time = measure_inference_time(model, X_test[:1000])
print("Average inference time per batch:", avg_time, "seconds")

