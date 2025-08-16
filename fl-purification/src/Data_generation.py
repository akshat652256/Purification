import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchattacks import CW
import shutil
from utils.misc.Attacks import fgsm_attack,pgd_attack,carlini_attack
from models.Classifier.CNN import MNIST_CNN
from models.Defensive_models.AE import DetectorIReformer,DetectorII
from dataloader import MNISTTopoDataset
import numpy as np


def get_dataloader_MNIST(batch_size=64, download=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST with normalization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Create train and validation splits (e.g., 90% train, 10% val)
    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform),
                             batch_size=64, shuffle=False)

    
    return train_loader, val_loader, test_loader


def load_model(model_name='mnist.pth', device='cpu'):
    """
    Load the specified pretrained classifier model from given path.

    Parameters:
    model_name : str
        Name of the model file (e.g., 'mnist.pth', 'mnist_AE1.pth', 'mnist_AE2.pth').
    device : str
        'cpu' or 'cuda' depending on device usage.

    Returns:
    model : torch.nn.Module
        Loaded and ready-to-evaluate model.
    """

    model_path = f'/kaggle/input/classifiers/Pretrained_classifiers/{model_name}'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model '{model_name}' not found at {model_path}")

    # Select architecture depending on the file
    if "AE1" in model_name:
        model = DetectorIReformer()  # Replace with your AE1 architecture
    elif "AE2" in model_name:
        model = DetectorII()  # Replace with your AE2 architecture
    else:
        model = MNIST_CNN()  # Standard MNIST model

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def generate_perturbed_full_loader(model, dataloader, attack_type='cw', device='cuda', **attack_params):
    """
    Generate a DataLoader containing adversarial examples for the entire dataset.

    Args:
        model: Trained model.
        dataloader: DataLoader for the entire dataset.
        attack_type: 'cw', 'fgsm' or 'pgd'.
        device: Device for computation ('cuda' or 'cpu').
        attack_params: Parameters for the attacks.

    Returns:
        perturbed_loader: DataLoader with adversarial examples and labels for the entire data.
    """
    model = model.to(device)
    model.eval()

    perturbed_images = []
    perturbed_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if attack_type == 'fgsm':
            inputs.requires_grad = True
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = inputs.grad.data
            epsilon = attack_params.get('epsilon', 0.1)
            adv_examples = fgsm_attack(inputs, epsilon, data_grad)
        elif attack_type == 'pgd':
            epsilon = attack_params.get('epsilon', 0.1)
            alpha = attack_params.get('alpha', 0.01)
            iters = attack_params.get('iters', 40)
            adv_examples = pgd_attack(model, inputs, labels, epsilon, alpha, iters)
        elif attack_type == 'cw':
            c = attack_params.get('c', 1)
            lr = attack_params.get('lr', 0.01)
            steps = attack_params.get('steps', 1000)
            kappa = attack_params.get('kappa', 0)
            attack = CW(model, c=c, lr=lr, steps=steps, kappa=kappa)
            adv_examples = attack(inputs, labels)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        perturbed_images.append(adv_examples.cpu())
        perturbed_labels.append(labels.cpu())

    perturbed_images = torch.cat(perturbed_images, dim=0)
    perturbed_labels = torch.cat(perturbed_labels, dim=0)

    perturbed_dataset = TensorDataset(perturbed_images, perturbed_labels)
    perturbed_loader = DataLoader(perturbed_dataset, batch_size=dataloader.batch_size, shuffle=False)

    return perturbed_loader


def save_perturbed_dataset(perturbed_loader, base_dir, dataset_name, attack_type, strength=None):
    """
    Save images and labels from the perturbed loader to the specified directory structure.
    """
    attack_folder = attack_type if not strength else f"{attack_type} {strength}"
    dir_path = os.path.join("/kaggle/working",base_dir, dataset_name, attack_folder)
    os.makedirs(dir_path, exist_ok=True)

    for batch_idx, (images, labels) in enumerate(perturbed_loader):
        save_path = os.path.join(dir_path, f"batch_{batch_idx}.pt")
        torch.save({'images': images, 'labels': labels}, save_path)

    print(f"Saved perturbed dataset under: {dir_path}")
    return dir_path

def zip_directory(dir_path, zip_output_path):
    """
    Zips the entire contents of dir_path into a zip file at zip_output_path.

    Args:
        dir_path (str): Directory to zip.
        zip_output_path (str): Full path including filename where zip will be saved.

    Returns:
        str: Path to the created zip file.
    """
    base_name = zip_output_path.replace('.zip', '')
    shutil.make_archive(base_name=base_name, format='zip', root_dir=dir_path)
    print(f"Files zipped successfully to {zip_output_path}")
    return zip_output_path


def save_model_path(model, model_type):
    base_dir = '/kaggle/working/trained_models'
    save_dir = os.path.join(base_dir, model_type.capitalize())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


