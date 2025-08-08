import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from trainer import train_classifier, train_detector, train_reformer, train_reformer_hiprnet
from models.Classifier.Resnet import BasicBlock , ResNet18_MedMNIST
from models.Detector.AE import SimpleAutoencoder
from models.Reformer.DAE import DenoisingAutoEncoder
from models.Reformer.Hypernet import AdaptiveLaplacianPyramidUNet
from Data_generation import get_dataloaders
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def save_model_path(model, model_type):
    base_dir = '/kaggle/working/trained_models'
    save_dir = os.path.join(base_dir, model_type.capitalize())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a model on MedMNIST dataset")
    parser.add_argument('--dataset', type=str, default='bloodmnist', help="Dataset name from MedMNIST")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument('--reformer-type', type=str, default="dae", help="set reformer to use")
    parser.add_argument('--model', type=str, default='classifier', choices=['classifier', 'detector', 'reformer'],
                        help="Choose which model training function to use")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Here you need to instantiate your model according to the chosen model type
    # For example:
    if args.model == 'classifier':
        model = ResNet18_MedMNIST()  
        train_func = train_classifier
    elif args.model == 'detector':
        model = SimpleAutoencoder()  
        train_func = train_detector
    elif args.model == 'reformer' and args.reformer_type == "dae":
        model = DenoisingAutoEncoder()  
        train_func = train_reformer
    elif args.model == 'reformer' and args.reformer_type == "hiprnet":
        model = AdaptiveLaplacianPyramidUNet()
        train_func = train_reformer_hiprnet # have to make modifications to this or create a new function for hiprnet related trianers

    # Train the model
    trained_model = train_func(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    save_model_path(trained_model, args.model)

if __name__ == "__main__":
    main()
