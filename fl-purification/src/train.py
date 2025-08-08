import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from trainer import train_classifier, train_detector, train_reformer  
from models.Classifier.Resnet import BasicBlock , ResNet18_MedMNIST
from models.Detector.AE import SimpleAutoencoder
from models.Reformer.DAE import DenoisingAutoEncoder

def get_dataloaders(data_flag, batch_size=64, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataClass(split='train', transform=transform, download=download)
    val_dataset = DataClass(split='val', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

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
        model = ResNet18_MedMNIST()  # Replace with your classifier model init
        train_func = train_classifier
    elif args.model == 'detector':
        model = SimpleAutoencoder()  # Replace with your detector model init
        train_func = train_detector
    elif args.model == 'reformer':
        model = DenoisingAutoEncoder()  # Replace with your reformer model init
        train_func = train_reformer

    # Train the model
    trained_model = train_func(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
