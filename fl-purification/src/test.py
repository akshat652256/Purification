import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import load_model_from_path
from Data_generation import get_dataloader_MNIST,load_model
from utils.misc.metrics import *
from dataloader import AdversarialDataset
from sklearn.metrics import f1_score  



def parse_args():
    parser = argparse.ArgumentParser(description="Filter adversarial images with recon threshold")
    parser.add_argument('--base_dir', type=str, default='medmnist', choices=['medmnist', 'others'],
                        help='Base directory (medmnist or others)')
    parser.add_argument('--dataset', type=str, default='bloodmnist', help='Dataset name')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--strength', type=str, default='strong', choices=['weak', 'strong', None],
                        help="Attack strength")
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    classifier_model = load_model(model_name='mnist.pth',device=device)
    detector_1 = load_model(model_name='mnist_AE1.pth',device=device)
    detector_2 = load_model(model_name='mnist_AE2.pth',device=device)
    reformer = load_model(model_name='mnist_AE1.pth',device=device)

    # Load clean dataloaders
    train_loader, val_loader, test_loader = get_dataloader_MNIST()

    # Compute JSD threshold on clean validation data using detector reconstructions
    threshold_1,threshold_2 = compute_threshold(detector_1,detector_2,val_loader,device=device)
    print(f"Computed JSD threshold: {threshold_1},{threshold_2}")

    # Create adversarial dataset instance with parameters from parser args
    adversarial_dataset = AdversarialDataset(
        base_dir=args.base_dir,
        dataset_name=args.dataset,
        attack_type=args.attack_type,
        strength=args.strength
    )
    
    adversarial_loader = get_adversarial_dataloader(adversarial_dataset)

    # Filter adversarial images based on JSD threshold
    filtered_loader = filter(detector_1,threshold_1,detector_2,threshold_2,adversarial_loader,device=device)

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
    reconstructed_loader = reconstruct_with_reformer(reformer, adversarial_loader, device=device)
    classify_images(classifier_model,reconstructed_loader,device=device)

    # 4) Full pipeline
    if filtered_loader is not None and len(filtered_loader.dataset) > 0:
        print(f"Full pipeline")
        reconstructed_loader = reconstruct_with_reformer(reformer, filtered_loader, device=device)
        classify_images(classifier_model, reconstructed_loader, device=device)



if __name__ == "__main__":
    main()