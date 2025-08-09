import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import load_model_from_path
from Data_generation import get_dataloaders
from utils.misc.metrics import *
from dataloader import AdversarialDataset
from sklearn.metrics import f1_score  


def parse_args():
    parser = argparse.ArgumentParser(description="Filter adversarial images with JSD threshold")
    parser.add_argument('--base_dir', type=str, default='medmnist', choices=['medmnist', 'others'],
                        help='Base directory (medmnist or others)')
    parser.add_argument('--dataset', type=str, default='bloodmnist', help='Dataset name from MedMNIST')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--strength', type=str, default='strong', choices=['weak', 'strong', None],
                        help="Attack strength")
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    classifier_model = load_model_from_path('classifier', device)
    detector_model = load_model_from_path('detector', device)
    reformer_model = load_model_from_path('reformer', device)

    # Load clean dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Compute JSD threshold
    jsd_threshold = compute_jsd_threshold(detector_model, val_loader, device=device)
    print(f"Computed JSD threshold: {jsd_threshold}")

    # Create adversarial dataset
    adversarial_dataset = AdversarialDataset(
        base_dir=args.base_dir,
        dataset_name=args.dataset,
        attack_type=args.attack_type,
        strength=args.strength
    )
    adversarial_loader = DataLoader(adversarial_dataset, batch_size=1, shuffle=False)

    # 1) Just send adversarial dataset directly to classifier
    classify_dataset(classifier_model, adversarial_loader, device=device, label_name='Raw Adversarial')

    # 2) Filter adversarial dataset using detector and pass to classifier
    filtered_loader = filter_adversarial_images_by_jsd(detector_model, adversarial_dataset, jsd_threshold, device=device)
    if filtered_loader is not None:
        classify_dataset(classifier_model, filtered_loader, device=device, label_name='Filtered-Adversarial')
    else:
        print("No images passed the JSD threshold filtering.")

    # 3) Pass adversarial dataset to reformer and pass to classifier
    recon_loader = pass_through_reformer(reformer_model, adversarial_loader, device=device)
    classify_dataset(classifier_model, recon_loader, device=device, label_name='Reformed-Adversarial')

    # 4) Filter, pass to reformer, then to classifier (current pipeline)
    if filtered_loader is not None:
        filtered_recon_loader = pass_through_reformer(reformer_model, filtered_loader, device=device)
        classify_dataset(classifier_model, filtered_recon_loader, device=device, label_name='Filtered+Reformed-Adversarial')
    else:
        print("No images passed the JSD threshold filtering.")


if __name__ == "__main__":
    main()
