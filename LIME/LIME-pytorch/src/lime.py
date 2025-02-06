import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LocalLinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LocalLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_local_model(original_instance, perturbed_instances, model, weights):
    labels = model(perturbed_instances).detach()
    local_model = LocalLinearModel(input_dim=original_instance.shape[0])
    optimizer = optim.Adam(local_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = local_model(perturbed_instances).squeeze()
        loss = torch.mean(weights * criterion(outputs, labels))
        loss.backward()
        optimizer.step()
    
    return local_model

def plot_feature_importance(local_model):
    feature_importance = local_model.linear.weight.detach().numpy().flatten()
    plt.bar(range(len(feature_importance)), np.abs(feature_importance))
    plt.xlabel("Feature Index")
    plt.ylabel("Importance (Absolute Coefficients)")
    plt.title("LIME Feature Importance")
    plt.savefig("results/feature_importance.png")
    plt.show()