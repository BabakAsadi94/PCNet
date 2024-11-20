# scripts/inference.py

import argparse
import torch
from torchvision import transforms
from PIL import Image
import yaml
import os

from models import AsphaltNetDualBranch
from utils.helpers import set_seed

def load_model(config, model_path, device):
    model = AsphaltNetDualBranch(
        num_classes=config['model']['num_classes'],
        feature_dim_cnn=config['model']['feature_dim_cnn'],
        feature_dim_transformer=config['model']['feature_dim_transformer'],
        common_dim=config['model']['common_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_bottom, image_top, device):
    with torch.no_grad():
        image_bottom = image_bottom.to(device)
        image_top = image_top.to(device)
        output = model(image_bottom, image_top)
    return output.item()

def main(config_path, model_path, image_bottom_path, image_top_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility (optional for inference)
    set_seed(config['training']['seed'])
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(config, model_path, device)
    
    # Preprocess images
    image_size = config['data']['image_size']
    image_bottom = preprocess_image(image_bottom_path, image_size)
    image_top = preprocess_image(image_top_path, image_size)
    
    # Make prediction
    predicted_ductility = predict(model, image_bottom, image_top, device)
    
    # Reverse scaling if necessary
    dataset = None  # If you have access to the dataset object, initialize it to reverse scaling
    # predicted_ductility = dataset.reverse_scaling(predicted_ductility)
    
    print(f"Predicted Ductility: {predicted_ductility:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Asphalt Ductility')
    parser.add_argument('--config', type=str, required=True, help='Path to config file (YAML)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--image_bottom', type=str, required=True, help='Path to bottom image')
    parser.add_argument('--image_top', type=str, required=True, help='Path to top image')
    args = parser.parse_args()
    
    main(args.config, args.model_path, args.image_bottom, args.image_top)
