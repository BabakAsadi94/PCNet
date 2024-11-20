# utils/compute_stats.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # For progress bar

from models.dual_branch import LDCFeatureExtractor  # Adjust the import path if necessary
from models.dual_branch import AsphaltDataset  # Assuming AsphaltDataset is defined in models.dual_branch
# If AsphaltDataset is defined elsewhere, adjust the import accordingly

def compute_mean_std(dataset, batch_size=32, num_workers=4):
    """
    Computes the mean and standard deviation of a dataset.

    Args:
        dataset (Dataset): PyTorch Dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: Mean and standard deviation per channel.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _, _ in tqdm(loader, desc="Computing Mean and Std"):
        batch_samples = images.size(0)  # Batch size (number of images)
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (B, C, H*W)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

def main():
    # Paths to your CSV files and image directories
    train_csv = os.path.join('data', 'train.csv')       # Replace with your training CSV file path
    train_images_dir = os.path.join('data', 'images', 'Training_images')  # Replace with your training images directory

    # Define transformations: Same as training but without normalization
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize images to 384x384
        transforms.ToTensor(),          # Convert image to tensor
        # Note: No normalization here
    ])

    # Initialize the training dataset
    train_dataset = AsphaltDataset(
        csv_file=train_csv,
        root_dir=train_images_dir,
        transform=transform,
        scale_target=False  # No scaling needed for mean/std computation
    )

    # Compute mean and std
    mean, std = compute_mean_std(train_dataset, batch_size=32, num_workers=4)

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    # Save the computed mean and std for later use
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    os.makedirs('data/stats', exist_ok=True)
    torch.save(stats, os.path.join('data', 'stats', 'train_mean_std.pth'))
    print("Mean and Std saved to 'data/stats/train_mean_std.pth'.")

if __name__ == "__main__":
    main()
