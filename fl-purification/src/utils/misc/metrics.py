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


def compute_jsd_threshold(detector_model, dataloader, device='cpu'):
    import torch
    detector_model.to(device)
    detector_model.eval()
    jsd_values = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructions = detector_model(images)
            
            # Compute JSD for each image in the batch
            batch_jsd = jsd(images, reconstructions)  # Assuming jsd function is defined elsewhere
            
            jsd_values.append(batch_jsd.cpu())

    all_jsd_values = torch.cat(jsd_values)
    avg_jsd = all_jsd_values.mean().item()
    return avg_jsd


def filter_adversarial_images_by_jsd(detector_model, adversarial_loader, jsd_threshold, device='cpu'):
    detector_model.to(device)
    detector_model.eval()
    
    kept_images = []
    kept_labels = []
    
    with torch.no_grad():
        for images, labels in adversarial_loader:
            images = images.to(device)
            # Get reconstructions from the detector model
            reconstructions = detector_model(images)
            
            # Calculate JSD for each image in the batch
            # Assuming images and reconstructions are normalized tensors with same shape
            batch_jsd = jsd(images, reconstructions)  # Expected shape [batch_size]
            batch_jsd = batch_jsd.view(batch_jsd.size(0), -1).mean(dim=1)  # [batch_size]
            # Filter images based on JSD threshold
            mask = batch_jsd <= jsd_threshold
            
            # Collect images and labels that pass the filter
            if mask.any():
                kept_images.append(images[mask].cpu())
                kept_labels.append(labels[mask].cpu())
    
    if len(kept_images) == 0:
        # No images passed the filter, return empty loader
        empty_dataset = TensorDataset(torch.empty((0, *images.shape[1:])), torch.empty(0, dtype=torch.long))
        return DataLoader(empty_dataset, batch_size=adversarial_loader.batch_size, shuffle=False)
    
    # Concatenate all kept images and labels
    filtered_images = torch.cat(kept_images, dim=0)
    filtered_labels = torch.cat(kept_labels, dim=0)
    
    # Create a new DataLoader from filtered data
    filtered_dataset = TensorDataset(filtered_images, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=adversarial_loader.batch_size, shuffle=False)
    
    return filtered_loader


def reconstruct_with_reformer(reformer_model, filtered_loader, device='cpu'):
    import torch
    reformer_model.to(device)
    reformer_model.eval()
    reconstructed_images = []
    with torch.no_grad():
        for images, _ in filtered_loader:
            images = images.to(device)
            # Pass images through the Reformer model
            outputs = reformer_model(images)
            # Collect reconstructed outputs on CPU
            reconstructed_images.append(outputs.cpu())

    # Concatenate all reconstructed image tensors
    reconstructed_tensor = torch.cat(reconstructed_images, dim=0)

    # Create a new DataLoader from the reconstructed images tensor
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(reconstructed_tensor)
    reconstructed_loader = DataLoader(dataset, batch_size=filtered_loader.batch_size, shuffle=False)

    return reconstructed_loader

from sklearn.metrics import f1_score

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
            
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"F1 score of images classification: {f1:.4f}")
    return f1
    