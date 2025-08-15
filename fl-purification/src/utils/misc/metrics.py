import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score

# images: input tensor (batch, 3, 28, 28)
# outputs: reformed tensor from your model (batch, 3, 28, 28)
# Assume both are in [0,1] range

    

def jsd(P, Q, eps=1e-12):
    # Normalize to probabilities (L1 norm)
    _P = P / torch.norm(P, p=1, dim=-1, keepdim=True)
    _Q = Q / torch.norm(Q, p=1, dim=-1, keepdim=True)

    # Mean distribution
    _M = 0.5 * (_P + _Q)

    # Clamp to avoid log(0)
    _P = torch.clamp(_P, eps, 1.0)
    _Q = torch.clamp(_Q, eps, 1.0)
    _M = torch.clamp(_M, eps, 1.0)

    # Compute JSD using KL Divergence
    jsd = 0.5 * (torch.sum(_P * torch.log(_P / _M), dim=-1) +
                 torch.sum(_Q * torch.log(_Q / _M), dim=-1))
    return jsd


def softmax_with_temperature(logits, T):
    return F.softmax(logits / T, dim=-1)


def get_adversarial_dataloader(adversarial_dataset, batch_size=64, shuffle=False):
    # Concatenate all batches into single tensors
    all_images = []
    all_labels = []
    for batch_data in adversarial_dataset.data:
        all_images.append(batch_data['images'])
        all_labels.append(batch_data['labels'])
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Create a flat TensorDataset of individual samples
    flat_dataset = TensorDataset(all_images, all_labels)

    # Return DataLoader with desired batch size
    loader = DataLoader(flat_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


