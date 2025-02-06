import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model import SimpleMLP
from src.utils import generate_perturbations, compute_weights
from src.lime import train_local_model, plot_feature_importance

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


model = SimpleMLP(input_dim=20)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

smaple_idx = 0
sample = X_test[smaple_idx].numpy()

perturbations = generate_perturbations(sample)
weights = compute_weights(sample, perturbations.numpy())

local_model = train_local_model(sample, perturbations, model, weights)

plot_feature_importance(local_model)