import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import load_model_from_path
from Data_generation import get_dataloaders,load_classifier,load_mnist_model
from utils.misc.metrics import *
from dataloader import AdversarialDataset
from sklearn.metrics import f1_score  
from utils.misc.metrics import get_adversarial_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Filter adversarial images with JSD threshold")
    parser.add_argument('--base_dir', type=str, default='medmnist', choices=['medmnist', 'others'],
                        help='Base directory (medmnist or others)')
    parser.add_argument('--dataset', type=str, default='bloodmnist', choices=['bloodmnist','mnist'],
                        help='Dataset name from MedMNIST')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--strength', type=str, default='strong', choices=['weak', 'strong', None],
                        help="Attack strength")
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.base_dir == 'medmnist':
        # Load classifier and detector models for MedMNIST
        classifier_model = load_classifier(args.base_dir,args.dataset, device=device)
        detector_model = load_model_from_path(model_type='detector', device=device)
        reformer_model = load_model_from_path(model_type='detector', device=device)
    else:
        # Load classifier and detector models for other datasets
        classifier_model = load_mnist_model(model_type='classifier',device=device)
        detector_model1 = load_mnist_model(model_type='detector1', device=device)
        detector_model2 = load_mnist_model(model_type='detector2', device=device)
        reformer_model = load_mnist_model(model_type='reformer', device=device)

        

    # Load clean dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Compute JSD threshold on clean validation data using detector reconstructions
    if args.base_dir == 'medmnist':
        threshold_1,threshold_2,threshold_3 = compute_thresholds(classifier_model,detector_model, val_loader, device=device)
        print(f"Computed L1 threshold: {threshold_1}")
        print(f"Computed jsd(T=10) threshold: {threshold_2}")
        print(f"Computed jsd(T=40) threshold: {threshold_3}")

    elif args.base_dir == 'others':
        threshold_1,threshold_2 = compute_thresholds_mnist(detector_model1,detector_model2,val_loader,device=device)

    # Create adversarial dataset instance with parameters from parser args
    adversarial_dataset = AdversarialDataset(
        base_dir= 'medmnist',
        dataset_name=args.dataset,
        attack_type=args.attack_type,
        strength=args.strength
    )
    
    adversarial_loader = get_adversarial_dataloader(adversarial_dataset)

    # Filter adversarial images based on JSD threshold
    if args.base_dir == 'medmnist':
        filtered_loader = filter(classifier_model,detector_model,adversarial_loader,threshold_1,threshold_2,threshold_3,device=device)
    elif args.base_dir == 'others':
        filtered_loader = filter_mnist(detector_model1,threshold_1,detector_model2,threshold_2,adversarial_loader,device=device)

    if filtered_loader is not None:
        print(f"Number of images passing through detector: {len(filtered_loader.dataset)}")

    # 1 None pipeline
    print(f"None pipeline")
    classify_images(classifier_model, adversarial_loader, device = device)

    # 2 Detector only pipeline
    if filtered_loader is not None and len(filtered_loader.dataset) > 0:
        print("Detector pipeline")
        classify_images(classifier_model, filtered_loader, device=device)

    # 3 Reformer only pipeline
    print(f"Reformer only pipeline")
    reconstructed_loader = reconstruct_with_reformer(reformer_model, adversarial_loader, device=device)
    classify_images(classifier_model,reconstructed_loader,device=device)

    # 4 Full pipeline
    if filtered_loader is not None and len(filtered_loader.dataset) > 0:
        print(f"Full pipeline")
        reconstructed_loader = reconstruct_with_reformer(reformer_model, filtered_loader, device=device)
        classify_images(classifier_model, reconstructed_loader, device=device)


    


if __name__ == "__main__":
    main()