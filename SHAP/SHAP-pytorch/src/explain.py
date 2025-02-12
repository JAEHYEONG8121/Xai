import torch
import numpy as np
import matplotlib.pyplot as plt
from src.shap import SHAPExplainer

def plot_shape_values(shap_values):
    feature_index = np.arange(len(shap_values.numpy()))
    plt.bar(feature_index, shap_values.numpy(), color="blue", alpha=0.6)
    plt.xlabel("Feature Index")
    plt.ylabel("SHAP Value")
    plt.title("SHAP Feature Importance")
    plt.savefig("results/feature_importance.png")
    plt.grid(True)
    plt.show()

def explain_instance(model, instance):
    explainer = SHAPExplainer(model)
    print("Original Prediction:", model(instance).detach().numpy())
    shap_values = explainer.explain(instance)
    print("SHAP Values:", shap_values)
    plot_shape_values(shap_values)