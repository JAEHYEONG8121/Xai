# main.py
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model import MLP
from src.train import train_model
from src.explain import explain_instance
from src.utils import generate_perturbations
from src.shap import SHAPExplainer

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

model = train_model(X_train, y_train, input_dim=20)

dataset_mean = X_train.mean(dim=0)

explainer = SHAPExplainer(model, dataset_mean=dataset_mean)
sample_instance = X_test[0]
explain_instance(model, sample_instance)
