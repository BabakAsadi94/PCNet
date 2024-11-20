import os
from PIL import Image
import matplotlib.pyplot as plt

# Set your directory paths
directory = r"C:\Users\jamal\OneDrive\Desktop\poker chip\PROJECT\data\All"
output_path = r"C:\Users\jamal\OneDrive\Desktop\poker chip\PROJECT\image_grid.png"

# Gather all images ending with '-B'
images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith('-B.jpg')]
images = images[:160]  # Select only the first 160 images to fit in the 10x16 grid

# Create a grid of 10 rows and 16 columns
fig, axes = plt.subplots(10, 16, figsize=(20, 12))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    img = Image.open(images[i])
    ax.imshow(img)
    ax.axis('off')  # Hide axis

# Save the resulting image
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

print(f"Grid image saved to {output_path}")
