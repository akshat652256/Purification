import os
import torch
from torch.utils.data import Dataset, DataLoader

class AdversarialDataset(Dataset):
    def __init__(self, base_dir, dataset_name, attack_type, strength=None):
        
        # Set the Kaggle input root - fixed dataset root folder in Kaggle environment
        kaggle_input_root = '/kaggle/input/purification'  # Set your Kaggle dataset root here

        attack_folder = attack_type if not strength else f"{attack_type} {strength}"

        # Compose the full directory path by joining Kaggle root with the rest
        self.dir_path = os.path.join(kaggle_input_root, base_dir, dataset_name, attack_folder)

        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        # List all .pt batch files sorted by batch index
        self.batch_files = sorted([
            f for f in os.listdir(self.dir_path)
            if f.endswith('.pt')
        ])

        self.data = []
        for batch_file in self.batch_files:
            batch_path = os.path.join(self.dir_path, batch_file)
            batch_data = torch.load(batch_path)  # dict with 'images' and 'labels'
            self.data.append(batch_data)

        self.num_batches = len(self.data)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_batches:
            raise IndexError(f"Index {idx} out of range for {self.num_batches} batches")

        batch_data = self.data[idx]
        images = batch_data['images']
        labels = batch_data['labels']

        return images, labels

class MNISTTopoDataset(Dataset):
    def __init__(self, clean_images, topo_images, labels, latents):
        self.clean_images = clean_images
        self.topo_images = topo_images
        self.labels = labels
        self.latents = latents

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean = torch.tensor(self.clean_images[idx], dtype=torch.float32)
        topo = torch.tensor(self.topo_images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        latent = torch.tensor(self.latents[idx], dtype=torch.float32)
        return clean, topo, label, latent