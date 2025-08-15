import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score

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


def compute_thresholds(classifier_model, detector_model, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    import collections
    detector_model.eval()
    classifier_model.eval()
    l1_sum = 0.0
    jsd_sum_T10 = 0.0
    jsd_sum_T40 = 0.0
    num_samples = 0

    # Use a defaultdict to track all JSD values for each class
    jsd_class_dict = collections.defaultdict(list)

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            recon_imgs = detector_model(imgs)

            # L1 reconstruction loss
            l1_loss = torch.abs(imgs - recon_imgs).mean(dim=(1,2,3))
            l1_sum += l1_loss.sum().item()
            batch_size = imgs.size(0)
            num_samples += batch_size

            # Classifier logits
            logits_original = classifier_model(imgs)
            logits_recon = classifier_model(recon_imgs)

            # Softmax with Temperature T=10 and T=40
            probs_orig_T10 = softmax_with_temperature(logits_original, T=10)
            probs_recon_T10 = softmax_with_temperature(logits_recon, T=10)
            jsd_T10 = jsd(probs_orig_T10, probs_recon_T10)  # shape: [batch_size]
            jsd_sum_T10 += jsd_T10.sum().item()

            # Track JSD values per class
            for i in range(batch_size):
                class_idx = targets[i].item()
                jsd_class_dict[class_idx].append(jsd_T10[i].item())

            # T=40 as usual
            probs_orig_T40 = softmax_with_temperature(logits_original, T=40)
            probs_recon_T40 = softmax_with_temperature(logits_recon, T=40)
            jsd_T40 = jsd(probs_orig_T40, probs_recon_T40)
            jsd_sum_T40 += jsd_T40.sum().item()

    # Print min/max for JSD T=10 for each class
    print("JSD T=10 ranges per class:")
    for class_idx in sorted(jsd_class_dict.keys()):
        values = jsd_class_dict[class_idx]
        print(f"Class {class_idx}: Min={min(values):.6f}, Max={max(values):.6f}, Count={len(values)}")

    threshold_1 = l1_sum / num_samples
    threshold_2 = jsd_sum_T10 / num_samples
    threshold_3 = jsd_sum_T40 / num_samples

    return threshold_1, threshold_2, threshold_3


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
            
            # L1 loss per image
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

            # Image filter mask: True if all below threshold
            mask = (l1 <= threshold_1) & (jsd_T10 <= threshold_2) & (jsd_T40 <= threshold_3)

            kept_images.append(imgs[mask].cpu())
            kept_targets.append(targets[mask].cpu())

    if len(kept_images) == 0:
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