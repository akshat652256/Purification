import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import load_model_from_path
from Data_generation import get_dataloaders,load_classifier
from utils.misc.metrics import *
from dataloader import AdversarialDataset
from sklearn.metrics import f1_score  
from utils.misc.metrics import get_adversarial_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Filter adversarial images with JSD threshold")
    parser.add_argument('--base_dir', type=str, default='medmnist', choices=['medmnist', 'others'],
                        help='Base directory (medmnist or others)')
    parser.add_argument('--dataset', type=str, default='bloodmnist', help='Dataset name from MedMNIST')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--strength', type=str, default='strong', choices=['weak', 'strong', None],
                        help="Attack strength")
    parser.add_argument('--reformer_type', type=str, default='dae', choices=['dae', 'hiprnet', 'laplacian'],
                        help="Type of reformer to use")
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    classifier_model = load_classifier(args.dataset, device=device)
    detector_model = load_model_from_path('detector', device=device)
    reformer_model = load_model_from_path('reformer', reformer_type=args.reformer_type, device=device)

    # Load clean dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Compute JSD threshold on clean validation data using detector reconstructions
    jsd_threshold = compute_jsd_threshold(detector_model,reformer_model,classifier_model, val_loader, device=device)
    print(f"Computed JSD threshold: {jsd_threshold}")

    # Create adversarial dataset instance with parameters from parser args
    adversarial_dataset = AdversarialDataset(
        base_dir=args.base_dir,
        dataset_name=args.dataset,
        attack_type=args.attack_type,
        strength=args.strength
    )
    
    adversarial_loader = get_adversarial_dataloader(adversarial_dataset)

    # Filter adversarial images based on JSD threshold
    filtered_loader = filter_adversarial_images_by_jsd(detector_model,classifier_model, adversarial_loader, jsd_threshold, device=device)

    if filtered_loader is not None:
        print(f"Number of images passing through detector: {len(filtered_loader.dataset)}")

    # 1) None pipeline
    print(f"None pipeline")
    classify_images(classifier_model, adversarial_loader, device = device)

    # 2) Detector only pipeline
    if filtered_loader is not None and len(filtered_loader.dataset) > 0:
        print("Detector pipeline")
        classify_images(classifier_model, filtered_loader, device=device)

    # 3) Reformer only pipeline
    print(f"Reformer only pipeline")
    reconstructed_loader = reconstruct_with_reformer(reformer_model, adversarial_loader, device=device)
    classify_images(classifier_model,reconstructed_loader,device=device)

    # 4) Full pipeline
    if filtered_loader is not None and len(filtered_loader.dataset) > 0:
        print(f"Full pipeline")
        reconstructed_loader = reconstruct_with_reformer(reformer_model, filtered_loader, device=device)
        classify_images(classifier_model, reconstructed_loader, device=device)


    


if __name__ == "__main__":
    main()