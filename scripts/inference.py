import argparse
import torch
from torchvision import transforms
from PIL import Image
import yaml
import os

# Import model and helper functions
from models.asphalt_net import AsphaltNetDualBranch
from utils.helpers import set_seed

def load_model(model_path, device):
    """ Load the trained model for inference. """
    model = AsphaltNetDualBranch(num_classes=1)  # Assuming the same architecture for both models
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """ Preprocess the input image for model inference. """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_bottom, image_top, device):
    """ Run inference and return the predicted value. """
    with torch.no_grad():
        image_bottom = image_bottom.to(device)
        image_top = image_top.to(device)
        output = model(image_bottom, image_top)
    return output.item()

def main(model_path, image_bottom_path, image_top_path):
    """ Main function to run inference """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess images
    image_bottom = preprocess_image(image_bottom_path).to(device)
    image_top = preprocess_image(image_top_path).to(device)
    
    # Make prediction
    prediction = predict(model, image_bottom, image_top, device)
    
    print(f"Predicted Value: {prediction:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Strength or Ductility using PCNet')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--image_bottom', type=str, required=True, help='Path to bottom image')
    parser.add_argument('--image_top', type=str, required=True, help='Path to top image')
    
    args = parser.parse_args()
    
    main(args.model_path, args.image_bottom, args.image_top)
