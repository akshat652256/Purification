import os
import numpy as np
import torch
from torch.utils.data import Dataset

class AdversarialDataset(Dataset):
    def __init__(self, base_dir, dataset_name, attack_type, strength=None, return_recon=False):
        """
        Loads adversarial dataset from a single .npz file instead of multiple .pt batches.

        Args:
            base_dir (str): Base directory (e.g., "medmnist", "Others")
            dataset_name (str): Dataset name (e.g., "bloodmnist", "mnist")
            attack_type (str): Attack type string ('fgsm', 'pgd', 'cw')
            strength (str or None): strength ('weak', 'strong')
            return_recon (bool): If True, return (orig, recon, label) instead of (orig, label)
        """
        # Kaggle dataset root
        kaggle_input_root = '/kaggle/working/output'

        # Build attack folder name
        attack_folder = attack_type if not strength else f"{attack_type} {strength}"

        # Expected filename format: adversarial_<dataset>_<attack_folder>_complete.npz
        npz_filename = f"adversarial_mnist_cw_strong_complete.npz"
        npz_path = os.path.join(kaggle_input_root, base_dir, dataset_name, npz_filename)

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load npz data
        npz_data = np.load(npz_path)
        self.original_images = npz_data["original_images"]
        self.labels = npz_data["labels"]
        self.return_recon = return_recon

        if return_recon:
            self.reconstructed_images = npz_data["reconstructed_images"]

    def __len__(self):
        return self.original_images.shape[0]

    def __getitem__(self, idx):
        orig = torch.tensor(self.original_images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.return_recon:
            recon = torch.tensor(self.reconstructed_images[idx], dtype=torch.float32)
            return orig, recon, label
        else:
            return orig, label
