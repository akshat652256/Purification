import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
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


def compute_jsd_threshold(detector_model, classifier_model, dataloader, device='cpu', temperature=2.0):
    detector_model.eval()
    classifier_model.eval()

    total_jsd = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack inputs (assuming dataloader returns (images, labels))
            x, _ = batch
            x = x.to(device)

            # Original logits & probabilities
            logits_x = classifier_model(x)  # shape: (B, num_classes)
            probs_x = softmax_with_temperature(logits_x, T=temperature)

            # Reconstructed input from autoencoder
            x_recon = detector_model(x)
            logits_recon = classifier_model(x_recon)
            probs_recon = softmax_with_temperature(logits_recon, T=temperature)

            # JSD for the batch (vector)
            batch_jsd = jsd(probs_x, probs_recon)  # shape: (B,)
            total_jsd += batch_jsd.sum().item()
            total_samples += x.size(0)

    avg_jsd = total_jsd / total_samples
    return avg_jsd


def filter_adversarial_images_by_jsd(detector_model, classifier_model, adversarial_loader, jsd_threshold, device='cpu', temperature=2.0):
    detector_model.eval()
    classifier_model.eval()
    detector_model.to(device)
    classifier_model.to(device)

    kept_images = []
    kept_labels = []

    with torch.no_grad():
        for images, labels in adversarial_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get classifier outputs for original images
            logits_x = classifier_model(images)
            probs_x = softmax_with_temperature(logits_x, T=temperature)

            # Get reconstructions from detector (autoencoder)
            reconstructions = detector_model(images)
            logits_recon = classifier_model(reconstructions)
            probs_recon = softmax_with_temperature(logits_recon, T=temperature)

            # Compute JSD for each sample in batch
            batch_jsd = jsd(probs_x, probs_recon)  # shape: (batch_size,)

            # Create mask for samples passing the threshold
            mask = batch_jsd <= jsd_threshold

            if mask.any():
                # Apply mask to both images and labels to keep them aligned
                filtered_images = images[mask].cpu()
                filtered_labels = labels[mask].cpu()

                kept_images.append(filtered_images)
                kept_labels.append(filtered_labels)

    if len(kept_images) == 0:
        # No images passed the filter, return empty loader
        empty_dataset = TensorDataset(torch.empty((0, *images.shape[1:])), torch.empty(0, dtype=torch.long))
        return DataLoader(empty_dataset, batch_size=adversarial_loader.batch_size, shuffle=False)

    # Concatenate filtered images and labels
    filtered_images = torch.cat(kept_images, dim=0)
    filtered_labels = torch.cat(kept_labels, dim=0)

    filtered_dataset = TensorDataset(filtered_images, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=adversarial_loader.batch_size, shuffle=False)

    return filtered_loader



def reconstruct_with_reformer(reformer_model, filtered_loader, device='cpu'):
    reformer_model.to(device)
    reformer_model.eval()
    
    reconstructed_images = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in filtered_loader:
            images = images.to(device)
            # Pass images through the Reformer model
            outputs = reformer_model(images)
            # Collect reconstructed outputs on CPU
            reconstructed_images.append(outputs.cpu())
            # Collect labels as well
            all_labels.append(labels.cpu())
    
    # Concatenate all reconstructed images and labels
    reconstructed_tensor = torch.cat(reconstructed_images, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    print(f"Reconstructed images shape: {reconstructed_tensor.shape}")
    print(f"Labels length: {labels_tensor.shape}")

    # Create a new DataLoader from the reconstructed images tensor and labels
    dataset = TensorDataset(reconstructed_tensor, labels_tensor)
    reconstructed_loader = DataLoader(dataset, batch_size=filtered_loader.batch_size, shuffle=False)
    
    return reconstructed_loader


def classify_images(classifier_model, perturbed_loader, device='cpu'):
    """
    Classifies images from the perturbed DataLoader and computes the macro F1 score.
    
    Args:
        classifier_model: The model used for classification.
        perturbed_loader: DataLoader yielding (images, labels) batches.
        device: Device to run computations ('cpu' or 'cuda').
        
    Returns:
        macro F1 score as a float.
    """
    classifier_model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in perturbed_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier_model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("Warning: No samples to classify, returning F1=0")
        return 0.0
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 score of images classification: {f1:.4f}")
    return f1
    

def get_one_tenth_loader(adversarial_loader):
    """
    Returns a DataLoader with roughly 1/10th of the data from adversarial_loader.
    """
    dataset = adversarial_loader.dataset
    total_size = len(dataset)
    target_size = max(1, total_size // 10)  # At least 1 sample

    # Just take the first `target_size` samples
    indices = list(range(target_size))

    subset = Subset(dataset, indices)
    reduced_loader = DataLoader(
        subset,
        batch_size=adversarial_loader.batch_size,
        shuffle=False  # keep order consistent
    )

    return reduced_loader