import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import f1_score

# images: input tensor (batch, 3, 28, 28)
# outputs: reformed tensor from your model (batch, 3, 28, 28)
# Assume both are in [0,1] range

def compute_psnr_ssim(images, outputs):
    images_np = images.cpu().numpy()
    outputs_np = outputs.cpu().numpy()
    n = images_np.shape[0]
    psnr_scores = []
    ssim_scores = []
    for i in range(n):
        x = images_np[i].transpose(1, 2, 0)
        y = outputs_np[i].transpose(1, 2, 0)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        # skimage wants shape (H, W, C) and float in [0,1]
        psnr_val = psnr(x, y, data_range=1.0)
        ssim_val = ssim(x, y, channel_axis=-1, data_range=1.0)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
    return np.mean(psnr_scores), np.mean(ssim_scores)
    

def jsd(p, q, eps=1e-6):
    m = 0.5 * (p + q)
    p = p + eps
    q = q + eps
    m = m + eps
    kld_pm = (p * (p / m).log()).sum(dim=1)
    kld_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kld_pm + kld_qm)

def compute_jsd_threshold(detector_model, dataloader, device='cpu'):
    detector_model.eval()
    jsd_scores = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = detector_model(images)
            # Flatten images and outputs to 2D tensors: (batch_size, features)
            images_flat = images.reshape(images.size(0), -1)
            outputs_flat = outputs.reshape(outputs.size(0), -1)
            # Normalize each sample to sum to 1 to form probability distributions
            images_prob = images_flat / images_flat.sum(dim=1, keepdim=True)
            outputs_prob = outputs_flat / outputs_flat.sum(dim=1, keepdim=True)
            # Calculate JSD between input and reconstruction for the batch
            batch_jsd = jsd(images_prob, outputs_prob)
            jsd_scores.append(batch_jsd.cpu().numpy())

    # Aggregate all batch JSD scores and compute the average
    jsd_scores = np.concatenate(jsd_scores)
    avg_jsd = np.mean(jsd_scores)
    return avg_jsd

def filter_adversarial_images_by_jsd(detector_model, adversarial_dataset, jsd_threshold, device='cpu'):
    detector_model.eval()
    filtered_images = []
    filtered_labels = []

    with torch.no_grad():
        for images, labels in adversarial_dataset:
            images = images.to(device)
            outputs = detector_model(images)

            # Flatten images and reconstructions to (batch_size, features)
            images_flat = images.reshape(images.size(0), -1)
            outputs_flat = outputs.reshape(outputs.size(0), -1)

            # Normalize to valid probability distributions
            images_prob = images_flat / images_flat.sum(dim=1, keepdim=True)
            outputs_prob = outputs_flat / outputs_flat.sum(dim=1, keepdim=True)

            # Compute JSD per sample
            batch_jsd = jsd(images_prob,outputs_prob)

            # Select indices where JSD is below or equal to the threshold
            keep_indices = (batch_jsd <= jsd_threshold).nonzero(as_tuple=True)[0]

            if len(keep_indices) > 0:
                filtered_images.append(images[keep_indices.cpu()].cpu())
                filtered_labels.append(labels[keep_indices.cpu()].cpu())

    if len(filtered_images) == 0:
        print("No images found below the JSD threshold.")
        return None

    # Concatenate all filtered images and labels
    all_images = torch.cat(filtered_images, dim=0)
    all_labels = torch.cat(filtered_labels, dim=0)

    # Use original adversarial batch size for filtered dataloader
    batch_size = adversarial_dataset.data[0]['images'].size(0)

    filtered_dataset = TensorDataset(all_images, all_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

    return filtered_loader


def classify_dataset(classifier_model, loader, device='cpu', label_name=''):
    classifier_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            print(f"Images shape before squeeze: {images.shape}")
            images = images.squeeze(0).to(device)  # Remove extra batch dim if present
            print(f"Images shape after squeeze: {images.shape}")
            labels = labels.squeeze(0).to(device)
            outputs = classifier_model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"[{label_name}] Classification F1 score: {f1:.4f}")
    return f1


def pass_through_reformer(reformer_model, loader, device='cpu'):
    reformer_model.eval()
    reconstructed = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.squeeze(0).to(device)  # Remove extra batch dimension if present
            labels = labels.squeeze(0).to(device)
            outputs = reformer_model(images)
            # Ensure outputs keep batch dimension
            if outputs.dim() == 3:  # If batch dimension is lost
                outputs = outputs.unsqueeze(0)
            reconstructed.append(outputs.cpu())
            labels_list.append(labels.cpu())
    all_recon = torch.cat(reconstructed, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    recon_dataset = TensorDataset(all_recon, all_labels)
    recon_loader = DataLoader(recon_dataset, batch_size=loader.batch_size, shuffle=False)
    return recon_loader

