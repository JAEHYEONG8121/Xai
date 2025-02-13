import numpy as np
import torch
from scipy.spatial.distance import euclidean

def generate_perturbations(instance, num_samples=500):
    perturbations = torch.bernoulli(torch.full((num_samples, instance.shape[0]), 0.5)) * instance
    return perturbations.clone().detach()

def compute_weights(original_instance, perturbed_instances, kernel_width=1.0):
    perturbed_instances = torch.tensor(perturbed_instances, dtype=torch.float32)
    distances = torch.cdist(original_instance.unsqueeze(0), perturbed_instances).squeeze(0)
    weights = torch.exp(- (distances ** 2) / (2 * kernel_width ** 2))
    return weights / weights.sum()

