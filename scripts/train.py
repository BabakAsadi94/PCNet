# scripts/train.py

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from models import AsphaltNetDualBranch
from data import AsphaltDataset
from utils.metrics import calculate_r2, calculate_rmse, calculate_mae, calculate_mape
from utils.visualization import (
    plot_loss,
    plot_r2,
    plot_rmse,
    plot_mape,
    plot_measured_vs_predicted
)
from utils.helpers import seconds_to_hms, set_seed
from utils.logger import get_logger
from torchvision import transforms

def evaluate_model(model, data_loader, criterion, dataset=None, device='cuda'):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        dataset (AsphaltDataset, optional): Dataset instance for reverse scaling. Defaults to None.
        device (torch.device, optional): Device to perform computation. Defaults to 'cuda'.

    Returns:
        tuple: (average_loss, r2, rmse, mae, mape, all_targets, all_outputs)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for bottom_images, top_images, targets in data_loader:
            bottom_images = bottom_images.to(device)
            top_images = top_images.to(device)
            targets = targets.to(device)

            outputs = model(bottom_images, top_images).squeeze()

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)

    if dataset and dataset.scale_target:
        # Reverse scale the targets and outputs
        all_targets = dataset.reverse_scaling(all_targets)
        all_outputs = dataset.reverse_scaling(all_outputs)

    # Calculate metrics
    r2 = calculate_r2(all_targets, all_outputs)
    rmse = calculate_rmse(all_targets, all_outputs)
    mae = calculate_mae(all_targets, all_outputs)
    mape = calculate_mape(all_targets, all_outputs)

    avg_loss = running_loss / len(data_loader)

    return avg_loss, r2, rmse, mae, mape, all_targets, all_outputs

def train_model(config):
    """
    Trains the AsphaltNetDualBranch model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary loaded from YAML file.
    """
    # Setup logging
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    logger = get_logger(log_file=os.path.join(config['training']['save_dir'], config['logging']['log_file']))
    logger.info("Starting training process.")

    # Set seeds for reproducibility
    set_seed(config['training']['seed'])

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(tuple(config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets
    train_dataset = AsphaltDataset(
        csv_file=config['data']['train_csv'],
        root_dir=config['data']['train_dir'],
        transform=train_transform,
        scale_target=True
    )

    test_dataset = AsphaltDataset(
        csv_file=config['data']['test_csv'],
        root_dir=config['data']['test_dir'],
        transform=test_transform,
        scale_target=True,
        min_val=train_dataset.min_val,
        max_val=train_dataset.max_val
    )

    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = AsphaltNetDualBranch(
        num_classes=config['model']['num_classes'],
        feature_dim_cnn=config['model']['feature_dim_cnn'],
        feature_dim_transformer=config['model']['feature_dim_transformer'],
        common_dim=config['model']['common_dim']
    )

    # Handle multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=5,
        verbose=True
    )

    # Initialize variables for tracking
    best_test_r2 = -float('inf')
    best_model_path = os.path.join(config['training']['save_dir'], f"{config['training']['save_path']}_best.pth")

    metrics = {
        'train_losses': [],
        'test_losses': [],
        'train_r2s': [],
        'test_r2s': [],
        'train_rmses': [],
        'test_rmses': [],
        'train_maes': [],
        'test_maes': [],
        'train_mapes': [],
        'test_mapes': []
    }

    start_time = time.time()

    # Training loop
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []

        for bottom_images, top_images, targets in train_loader:
            bottom_images = bottom_images.to(device)
            top_images = top_images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(bottom_images, top_images).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            all_train_targets.extend(targets.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().numpy())

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        if train_dataset.scale_target:
            all_train_targets = train_dataset.reverse_scaling(np.array(all_train_targets))
            all_train_outputs = train_dataset.reverse_scaling(np.array(all_train_outputs))

        train_r2 = calculate_r2(all_train_targets, all_train_outputs)
        train_rmse = calculate_rmse(all_train_targets, all_train_outputs)
        train_mae = calculate_mae(all_train_targets, all_train_outputs)
        train_mape = calculate_mape(all_train_targets, all_train_outputs)

        # Evaluate on test set
        test_loss, test_r2, test_rmse, test_mae, test_mape, _, _ = evaluate_model(
            model,
            test_loader,
            criterion,
            dataset=test_dataset,
            device=device
        )

        # Update scheduler
        scheduler.step(test_loss)

        # Save metrics
        metrics['train_losses'].append(train_loss)
        metrics['test_losses'].append(test_loss)
        metrics['train_r2s'].append(train_r2)
        metrics['test_r2s'].append(test_r2)
        metrics['train_rmses'].append(train_rmse)
        metrics['test_rmses'].append(test_rmse)
        metrics['train_maes'].append(train_mae)
        metrics['test_maes'].append(test_mae)
        metrics['train_mapes'].append(train_mape)
        metrics['test_mapes'].append(test_mape)

        # Check for best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Epoch {epoch}: Best model saved with Test R²: {test_r2:.4f}")

        # Log epoch metrics
        epoch_time = time.time() - start_time
        formatted_time = seconds_to_hms(epoch_time)
        logger.info(
            f"Epoch [{epoch}/{config['training']['epochs']}], "
            f"Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}, "
            f"Train MAE: {train_mae:.4f}, Train MAPE: {train_mape:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}, "
            f"Test MAE: {test_mae:.4f}, Test MAPE: {test_mape:.2f}%, "
            f"Time: {formatted_time}"
        )

    logger.info("Training complete.")

    # Plot metrics
    plot_loss(metrics['train_losses'], metrics['test_losses'], config['training']['epochs'], save_dir=config['training']['save_dir'])
    plot_r2(metrics['train_r2s'], metrics['test_r2s'], config['training']['epochs'], save_dir=config['training']['save_dir'])
    plot_rmse(metrics['train_rmses'], metrics['test_rmses'], config['training']['epochs'], save_dir=config['training']['save_dir'])
    plot_mape(metrics['train_mapes'], metrics['test_mapes'], config['training']['epochs'], save_dir=config['training']['save_dir'])

    # Evaluate best model
    best_model = AsphaltNetDualBranch(
        num_classes=config['model']['num_classes'],
        feature_dim_cnn=config['model']['feature_dim_cnn'],
        feature_dim_transformer=config['model']['feature_dim_transformer'],
        common_dim=config['model']['common_dim']
    )
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(best_model)
    best_model.to(device)
    best_model.eval()

    train_loss, train_r2, train_rmse, train_mae, train_mape, train_targets, train_outputs = evaluate_model(
        best_model,
        train_loader,
        criterion,
        dataset=train_dataset,
        device=device
    )
    test_loss, test_r2, test_rmse, test_mae, test_mape, test_targets, test_outputs = evaluate_model(
        best_model,
        test_loader,
        criterion,
        dataset=test_dataset,
        device=device
    )

    # Plot Measured vs Predicted
    plot_measured_vs_predicted(train_targets, train_outputs, test_targets, test_outputs, save_dir=config['training']['save_dir'])

    # Log final evaluation
    logger.info(
        f"Final Training Loss: {train_loss:.4f}, Final Training R²: {train_r2:.4f}, "
        f"Final Training RMSE: {train_rmse:.4f}, Final Training MAE: {train_mae:.4f}, "
        f"Final Training MAPE: {train_mape:.2f}%"
    )
    logger.info(
        f"Final Test Loss: {test_loss:.4f}, Final Test R²: {test_r2:.4f}, "
        f"Final Test RMSE: {test_rmse:.4f}, Final Test MAE: {test_mae:.4f}, "
        f"Final Test MAPE: {test_mape:.2f}%"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train AsphaltNetDualBranch Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_model(config)
