# data/data_asphalt.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from torchvision import transforms

class AsphaltDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, scale_target=True, min_val=None, max_val=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            scale_target (bool): Whether to apply Min-Max scaling to the target.
            min_val (float, optional): Minimum value for scaling (used for test set).
            max_val (float, optional): Maximum value for scaling (used for test set).
        """
        self.data_frame = pd.read_csv(csv_file)  # Load CSV file with Ductility values
        self.root_dir = root_dir
        self.transform = transform
        self.scale_target = scale_target

        if self.scale_target:
            if min_val is None or max_val is None:
                self.min_val = self.data_frame['Ductility'].min()
                self.max_val = self.data_frame['Ductility'].max()
            else:
                self.min_val = min_val
                self.max_val = max_val

            # Avoid division by zero
            if self.max_val - self.min_val != 0:
                self.data_frame['Ductility'] = (self.data_frame['Ductility'] - self.min_val) / (self.max_val - self.min_val)
            else:
                self.data_frame['Ductility'] = 0.0  # If all values are the same

    def __len__(self):
        return len(self.data_frame)

    def find_image(self, base_name, suffix):
        """Try to find the image with multiple possible extensions."""
        extensions = ['png', 'jpg', 'jpeg']  # List of supported extensions
        for ext in extensions:
            img_path = os.path.join(self.root_dir, f"{base_name}-{suffix}.{ext}")
            if os.path.exists(img_path):
                return img_path
        return None  # If no file found

    def __getitem__(self, idx):
        base_name = str(self.data_frame.iloc[idx, 0]).strip()  # Ensure base_name is string and strip whitespace

        # Check for both bottom and top images with possible extensions
        img_name_bottom = self.find_image(base_name, 'B')
        img_name_top = self.find_image(base_name, 'T')

        if img_name_bottom is None or img_name_top is None:
            print(f"Warning: Missing image(s) for {base_name}, skipping.")
            # Recursively try the next index (with a limit to prevent infinite recursion)
            next_idx = (idx + 1) % len(self.data_frame)
            if next_idx == idx:
                raise RuntimeError(f"No valid images found in the dataset.")
            return self.__getitem__(next_idx)

        # Load both bottom and top images
        try:
            image_bottom = Image.open(img_name_bottom).convert('RGB')
            image_top = Image.open(img_name_top).convert('RGB')
        except Exception as e:
            print(f"Error loading images for {base_name}: {e}")
            # Recursively try the next index
            next_idx = (idx + 1) % len(self.data_frame)
            if next_idx == idx:
                raise RuntimeError(f"No valid images found in the dataset.")
            return self.__getitem__(next_idx)

        # Apply transformations if specified
        if self.transform:
            image_bottom = self.transform(image_bottom)
            image_top = self.transform(image_top)

        # Get the target value (Ductility)
        ductility_value = float(self.data_frame.iloc[idx, 1])

        return image_bottom, image_top, torch.tensor(ductility_value, dtype=torch.float32)

    def reverse_scaling(self, value):
        """Reverse the Min-Max scaling to get the original Ductility value."""
        return value * (self.max_val - self.min_val) + self.min_val
