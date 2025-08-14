import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import f1_score


def compute_threshold(detector_1, detector_2, val_loader, device):
    detector_1.to(device)
    detector_2.to(device)
    detector_1.eval()
    detector_2.eval()

    l1_total = 0.0
    l2_total = 0.0
    count = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs_1 = detector_1(images)
            outputs_2 = detector_2(images)

            # L1 norm (mean absolute error) per sample
            l1_err = torch.mean(torch.abs(images - outputs_1), dim=[1, 2, 3])
            # L2 norm (mean squared error) per sample
            l2_err = torch.mean((images - outputs_2) ** 2, dim=[1, 2, 3])

            l1_total += l1_err.sum().item()
            l2_total += l2_err.sum().item()
            count += images.size(0)

    avg_l1 = l1_total / count
    avg_l2 = l2_total / count

    # Return both average L1 and average L2 reconstruction errors
    return avg_l1, avg_l2


def filter(detector_1, threshold_1, detector_2, threshold_2, adversarial_loader, device):
    detector_1.to(device)
    detector_2.to(device)
    detector_1.eval()
    detector_2.eval()

    accepted_images = []
    accepted_labels = []

    with torch.no_grad():
        for images, labels in adversarial_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs_1 = detector_1(images)
            outputs_2 = detector_2(images)

            # L1 reconstruction error per sample for detector_1
            l1_err = torch.mean(torch.abs(images - outputs_1), dim=[1, 2, 3])
            # L2 reconstruction error per sample for detector_2
            l2_err = torch.mean((images - outputs_2) ** 2, dim=[1, 2, 3])

            # Condition: errors below respective thresholds
            mask = (l1_err < threshold_1) & (l2_err < threshold_2)

            # Collect only accepted images and corresponding labels
            if mask.any():
                accepted_images.append(images[mask].cpu())
                accepted_labels.append(labels[mask].cpu())

    if len(accepted_images) == 0:
        print("No images passed the filtering criteria.")
        return None

    # Concatenate all accepted images and labels
    filtered_images = torch.cat(accepted_images)
    filtered_labels = torch.cat(accepted_labels)

    # Create a TensorDataset and DataLoader with the filtered subset
    filtered_dataset = TensorDataset(filtered_images, filtered_labels)
    filtered_loader = DataLoader(filtered_dataset, batch_size=adversarial_loader.batch_size, shuffle=False)

    return filtered_loader


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