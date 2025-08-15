import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from trainer import train_classifier, train_autoencoder
from models.Classifier.CNN import MEDMNIST_CNN
from models.Defensive_models.AE import MEDMNIST_AE
from Data_generation import get_dataloaders
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import numpy as np
import wandb

def save_model_path(model, model_type):
    base_dir = '/kaggle/working/trained_models'
    save_dir = os.path.join(base_dir, model_type.capitalize())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model_from_path(model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a trained model of the specified type from the predefined saved model path.

    Args:
        model_type (str): One of 'classifier', 'detector', 'reformer'.
        device (str): Device to map the model to ('cuda' or 'cpu').

    Returns:
        model: The loaded model with weights.
    """
    base_dir = '/kaggle/working/trained_models'
    model_dir = os.path.join(base_dir, model_type.capitalize())
    model_path = os.path.join(model_dir, f'{model_type}_model.pth')

    # Instantiate model architecture based on model_type
    if model_type == 'classifier':
        model = MEDMNIST_CNN()
    elif model_type == 'detector':
        model = MEDMNIST_AE()
    elif model_type == 'reformer':
        model = MEDMNIST_AE()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load saved weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded {model_type} model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train a model on MedMNIST dataset")
    parser.add_argument('--dataset', type=str, default='bloodmnist', help="Dataset name from MedMNIST")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument('--model', type=str, default='classifier', choices=['classifier', 'detector', 'reformer'],
                        help="Choose which model training function to use")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Here you need to instantiate your model according to the chosen model type
    # For example:
    if args.model == 'classifier':
        model = MEDMNIST_CNN()  
        train_func = train_classifier
    elif args.model == 'detector':
        model = MEDMNIST_AE() 
        train_func = train_autoencoder
    elif args.model == 'reformer':
        model = MEDMNIST_AE()  
        train_func = train_autoencoder
    

    # Train the model
    trained_model = train_func(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    save_model_path(trained_model, args.model)

if __name__ == "__main__":
    main()


