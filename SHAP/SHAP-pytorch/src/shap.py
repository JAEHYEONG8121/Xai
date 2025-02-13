import math
import torch
from itertools import combinations
from src.utils import generate_perturbations
from src.utils import compute_weights

class SHAPExplainer:
    def __init__(self, model, num_samples=500, kernel_width=1.0):
        self.model = model
        self.num_samples = num_samples
        self.kernel_width = kernel_width
    
    def explain(self, instance):
        num_features = instance.shape[0]
        shap_values = torch.zeros(num_features)

        perturbations = generate_perturbations(instance, self.num_samples)
        weights = compute_weights(instance, perturbations, self.kernel_width)

        
        f_x = self.model(instance.unsqueeze(0)).detach().squeeze()

        f_S = self.model(perturbations).detach()

        for i in range(num_features):
            perturbed_with_i = perturbations.clone()
            perturbed_with_i[:, i] = instance[i]

            f_S_i = self.model(perturbed_with_i).detach()

            shap_values[i] = torch.sum(weights * (f_S_i - f_S)) / torch.sum(weights)

        return shap_values