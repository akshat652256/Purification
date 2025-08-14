import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from trainer import train_classifier, train_autoencoder
from models.Classifier.CNN import MNIST_CNN
from models.Defensive_models.AE import DetectorIReformer,DetectorII
from Data_generation import get_dataloader_MNIST
import numpy as np
import wandb

def save_model_path(model, model_type):
    base_dir = '/kaggle/working/trained_models'
    save_dir = os.path.join(base_dir, model_type.capitalize())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model_from_path(model_type, detector_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        model = MNIST_CNN()
    elif model_type == 'detector' and detector_type == "D1":
        model = DetectorIReformer()
    elif model_type == 'detector' and detector_type == "D2":
        model = DetectorII()
    elif model_type == 'reformer':
        model = DetectorIReformer()
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
    parser.add_argument('--detector_type', type=str, default="D1", help="set detector to use")
    parser.add_argument('--model', type=str, default='classifier', choices=['classifier', 'detector', 'reformer'],
                        help="Choose which model training function to use")
    args = parser.parse_args()


    train_loader, val_loader, test_loader = get_dataloader_MNIST()

    # Here you need to instantiate your model according to the chosen model type
    # For example:
    if args.model == 'classifier':
        model = MNIST_CNN()  
        train_func = train_classifier
    elif args.model == 'detector' and args.detector_type == "D1":
        model = DetectorIReformer()  
        train_func = train_autoencoder
    elif args.model == 'detector' and args.detector_type == "D2":
        model = DetectorII()  
        train_func = train_autoencoder
    elif args.model == 'reformer':
        model = DetectorIReformer() 
        train_func = train_autoencoder

    # Train the model
    trained_model = train_func(model, train_loader, val_loader)
    save_model_path(trained_model, args.model)

if __name__ == "__main__":
    main()


