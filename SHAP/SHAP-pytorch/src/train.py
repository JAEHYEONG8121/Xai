import torch
from torch.optim import Adam
from torch.nn import BCELoss
from src.model import MLP

def train_model(X_train, y_train, input_dim, epochs=100, lr=0.01):
    model = MLP(input_dim)
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model
