import numpy as np
import torch
from scipy.spatial.distance import euclidean

def generate_perturbations(instance, num_samples=500, noise_level=0.1):
    perturbations = instance + noise_level * np.random.randn(num_samples, instance.shape[0])
    return torch.tensor(perturbations, dtype=torch.float32)

def compute_weights(original_instance, perturbed_instances, kernel_width=0.75):
    distances = np.array([euclidean(original_instance, p) for p in perturbed_instances])
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))
    return torch.tensor(weights, dtype=torch.float32)

