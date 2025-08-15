import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms


def train_autoencoder(model, train_loader, val_loader, epochs=200, lr=0.001,device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            # Add Gaussian noise
            noise = torch.randn_like(imgs) * 0.025
            noisy_imgs = torch.clamp(imgs + noise, 0.0, 1.0)
            output = model(noisy_imgs)
            loss = criterion(output, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = []
        val_ssim = []
        with torch.no_grad():
            for val_imgs, _ in val_loader:
                val_imgs = val_imgs.to(device)
                noise = torch.randn_like(val_imgs) * 0.025
                noisy_val_imgs = torch.clamp(val_imgs + noise, 0.0, 1.0)
                val_out = model(noisy_val_imgs)
                val_loss += criterion(val_out, val_imgs).item() * val_imgs.size(0)
                # Compute PSNR and SSIM using torchmetrics
                batch_psnr = psnr_metric(val_out, val_imgs)
                batch_ssim = ssim_metric(val_out, val_imgs)
                val_psnr.append(batch_psnr.item())
                val_ssim.append(batch_ssim.item())
        val_loss /= len(val_loader.dataset)
        avg_psnr = np.mean(val_psnr)
        avg_ssim = np.mean(val_ssim)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {avg_psnr:.4f} | Val SSIM: {avg_ssim:.4f}")

    return model


def train_classifier(model, train_loader, val_loader, epochs=350, lr=1e-2,device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            # Apply data augmentation on the fly
            images = transforms.RandomHorizontalFlip()(images)
            images = transforms.RandomAffine(degrees=0, translate=(0.1,0.1))(images)
            
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(labels, (list, tuple)):
                labels = labels[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = running_corrects.double() / total

        # Validate
        model.eval()
        val_loss, val_corrects, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(labels, (list, tuple)):
                    labels = labels[0]
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total

        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return model