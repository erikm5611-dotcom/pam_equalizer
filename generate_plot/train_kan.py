# train_kan.py
from kan import KAN
import torch

def build_kan(input_dim, hidden=16, grid=5, k=3, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN(width=[input_dim, hidden, 1], grid=grid, k=k, seed=seed, device=device).to(device)
    model.speed()
    return model