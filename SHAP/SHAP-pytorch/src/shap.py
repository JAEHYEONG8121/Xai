import math
import torch
from itertools import combinations
from src.utils import compute_weights

class SHAPExplainer:
    def __init__(self, model, num_samples=500, kernel_width=1.0, dataset_mean=None):
        self.model = model
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.dataset_mean = dataset_mean  

    def explain(self, instance):
        num_features = instance.shape[0]
        shap_values = torch.zeros(num_features)

        for i in range(num_features):
            subsets = list(combinations(range(num_features), i))

            for subset in subsets:
                S = torch.zeros_like(instance)
                S[list(subset)] = instance[list(subset)]

                f_S = self.model(S.unsqueeze(0)).detach().squeeze()
                f_S_i = self.model(instance.unsqueeze(0)).detach().squeeze()

                weight = self.shapley_weight(len(subset), num_features)
                shap_values[i] += weight * (f_S_i - f_S)

        return shap_values

    def shapley_weight(self, subset_size, num_features):
        return (math.factorial(subset_size) * 
                math.factorial(num_features - subset_size - 1)) / math.factorial(num_features)
