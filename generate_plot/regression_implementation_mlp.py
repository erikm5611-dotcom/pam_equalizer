# regression_implementation_mlp.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PAM_LEVELS = np.array([-3, -1, 1, 3], dtype=np.float32)

def standardize_fit(X): # this function computes the mean and std of the training data
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sigma

def standardize_apply(X, mu, sigma): # applies standardization
    return (X - mu) / sigma

def nearest_level(y_pred): # converts the continuous value from the regression model to the value in PAM that is closest to it
    # y_pred shape (N,)
    levels = PAM_LEVELS
    return levels[np.argmin(np.abs(y_pred[:, None] - levels[None, :]), axis=1)]

def pam4_to_bits(sym): # converts symbol to bit for PAM-4
    # Gray mapping
    if sym == -3: return (0, 0)
    if sym == -1: return (0, 1)
    if sym ==  1: return (1, 1)
    if sym ==  3: return (1, 0)
    raise ValueError(sym)

def compute_ber_ser(y_true_levels, y_hat_levels): # computes both the BER and SER goven true labels and predicted labels
    y_true_levels = y_true_levels.astype(int)
    y_hat_levels = y_hat_levels.astype(int)

    ser = np.mean(y_true_levels != y_hat_levels)

    bit_err = 0
    for sh, st in zip(y_hat_levels, y_true_levels):
        bh = pam4_to_bits(sh)
        bt = pam4_to_bits(st)
        bit_err += (bh[0] != bt[0]) + (bh[1] != bt[1])

    ber = bit_err / (2 * len(y_true_levels))
    return ber, ser

class MLP(nn.Module): # our simple MLP
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

def train_regressor_torch(model, Xtr, ytr, Xval, yval,                  # trains the regression model with early stopping based on value of validation loss
                          lr=1e-3, wd=1e-4, batch=256,
                          max_epochs=80, patience=10, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    tr_loader = DataLoader(TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    ), batch_size=batch, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(Xval, dtype=torch.float32),
        torch.tensor(yval, dtype=torch.float32).unsqueeze(1)
    ), batch_size=512, shuffle=False)

    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                tot += loss.item() * xb.size(0)
                n += xb.size(0)
        v = tot / n

        if v < best - 1e-4:
            best = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best

def predict_regressor(model, X, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        y = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    return y