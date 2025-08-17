import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score
from dataloader import MNISTTopoDataset

def jsd(P, Q, eps=1e-12):
    """
    Jensen-Shannon Divergence between two probability distributions P and Q.
    Both P and Q should be torch tensors with shape (..., num_classes).
    They should sum to 1 along the last dimension.
    """
    M = 0.5 * (P + Q)
    kl_pm = torch.sum(P * torch.log((P + eps) / (M + eps)), dim=-1)
    kl_qm = torch.sum(Q * torch.log((Q + eps) / (M + eps)), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

def softmax_with_temperature(logits, T):
    """
    Softmax function with temperature.
    logits: torch tensor of scores (..., num_classes)
    T: temperature scalar (float)
    """
    return F.softmax(logits / T, dim=-1)

def compute_thresholds_mnist(detector_1, detector_2, val_loader, device):
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

def filter_mnist(detector_1, threshold_1, detector_2, threshold_2, adversarial_loader, device):
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

def compute_thresholds(classifier_model, detector_model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    import collections
    detector_model.eval()
    classifier_model.eval()
    
    l1_values = []  # Collect L1 loss for all images (not per-class)
    jsd_T10_class_dict = collections.defaultdict(list)
    jsd_T40_class_dict = collections.defaultdict(list)
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            recon_imgs = detector_model(imgs)
            
            # L1 reconstruction loss per image
            l1_loss = torch.abs(imgs - recon_imgs).mean(dim=(1,2,3))  # shape: [batch_size]
            l1_values.extend(l1_loss.cpu().numpy().tolist())
            
            # Classifier logits
            logits_original = classifier_model(imgs)
            logits_recon = classifier_model(recon_imgs)
            
            # Softmax with Temperature T=10
            probs_orig_T10 = softmax_with_temperature(logits_original, T=10)
            probs_recon_T10 = softmax_with_temperature(logits_recon, T=10)
            jsd_T10 = jsd(probs_orig_T10, probs_recon_T10)  # shape: [batch_size]
            
            # Softmax with Temperature T=40
            probs_orig_T40 = softmax_with_temperature(logits_original, T=40)
            probs_recon_T40 = softmax_with_temperature(logits_recon, T=40)
            jsd_T40 = jsd(probs_orig_T40, probs_recon_T40)  # shape: [batch_size]
            
            # Track JSD values per class
            for i in range(imgs.size(0)):
                class_idx = targets[i].item()
                jsd_T10_class_dict[class_idx].append(jsd_T10[i].item())
                jsd_T40_class_dict[class_idx].append(jsd_T40[i].item())
    
    # Calculate thresholds
    threshold_1 = sum(l1_values) / len(l1_values) if l1_values else 0.0  # Single average for L1
    threshold_2 = {}
    threshold_3 = {}
    class_indices = sorted(set(list(jsd_T10_class_dict.keys()) + list(jsd_T40_class_dict.keys())))
    for class_idx in class_indices:
        threshold_2[class_idx] = sum(jsd_T10_class_dict[class_idx]) / len(jsd_T10_class_dict[class_idx]) if jsd_T10_class_dict[class_idx] else 0.0
        threshold_3[class_idx] = sum(jsd_T40_class_dict[class_idx]) / len(jsd_T40_class_dict[class_idx]) if jsd_T40_class_dict[class_idx] else 0.0

    return threshold_1, threshold_2, threshold_3



import torch
from torch.utils.data import TensorDataset, DataLoader

def filter(classifier_model, detector_model, adversarial_loader, threshold_1, threshold_2, threshold_3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    detector_model.eval()
    classifier_model.eval()
    kept_images = []
    kept_targets = []

    with torch.no_grad():
        for imgs, targets in adversarial_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Detector reconstruction
            recon_imgs = detector_model(imgs)
            
            # L1 loss per image (threshold_1 is a scalar)
            l1 = torch.abs(imgs - recon_imgs).mean(dim=(1,2,3))
            
            # Classifier logits
            logits_orig = classifier_model(imgs)
            logits_recon = classifier_model(recon_imgs)

            # JSD (T=10)
            probs_orig_T10 = softmax_with_temperature(logits_orig, T=10)
            probs_recon_T10 = softmax_with_temperature(logits_recon, T=10)
            jsd_T10 = jsd(probs_orig_T10, probs_recon_T10)
            
            # JSD (T=40)
            probs_orig_T40 = softmax_with_temperature(logits_orig, T=40)
            probs_recon_T40 = softmax_with_temperature(logits_recon, T=40)
            jsd_T40 = jsd(probs_orig_T40, probs_recon_T40)

            # Per-sample mask using scalar L1 threshold and per-class JSD thresholds
            batch_keep_mask = []
            for i in range(imgs.size(0)):
                class_idx = targets[i].item()
                t2 = threshold_2.get(class_idx, float('inf'))
                t3 = threshold_3.get(class_idx, float('inf'))
                keep = (l1[i] <= threshold_1) and (jsd_T10[i] <= t2) and (jsd_T40[i] <= t3)
                batch_keep_mask.append(keep)

            batch_keep_mask = torch.tensor(batch_keep_mask, dtype=torch.bool)
            kept_images.append(imgs[batch_keep_mask].cpu())
            kept_targets.append(targets[batch_keep_mask].cpu())

    if not kept_images or all(img.numel() == 0 for img in kept_images):
        print("No images passed the filtering criteria.")
        return None
    
    filtered_imgs = torch.cat(kept_images, dim=0)
    filtered_targets = torch.cat(kept_targets, dim=0)
    filtered_dataset = TensorDataset(filtered_imgs, filtered_targets)
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


def get_advmnist_topo_loaders(attack_type, strength, batch_size=64):
    """
    attack_type: e.g., 'cw', 'pgd', etc.
    strength: e.g., 'strong', 'weak', etc.
    Loads the appropriate npz file and returns a DataLoader.
    """
    base_path = "/kaggle/input/invi_mnist_64/pytorch/default/1/"
    file_name = f"adversarial_mnist_{attack_type}_{strength}_complete.npz"
    npz_path = base_path + file_name

    data = np.load(npz_path)
    clean_images = data['original_images']
    topo_images = data['reconstructed_images']
    labels = data['labels']
    latents = data['latents']
    
    full_dataset = MNISTTopoDataset(clean_images, topo_images, labels, latents)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    return full_loader


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
            labels = labels.squeeze()
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