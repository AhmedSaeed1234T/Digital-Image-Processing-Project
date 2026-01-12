# person2_preprocessing.py

from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# Load a single image
def load_image(path):
    """
    Load an image from the given file path.
    Returns a NumPy array in RGB format.
    """
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Convert image to grayscale
def to_gray(img_array):
    """
    Convert an RGB image array to grayscale.
    If the image is already grayscale, it returns as is.
    """
    if len(img_array.shape) == 3:  # RGB image
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return img_array

# Apply histogram equalization (grayscale)
def apply_histogram_equalization(gray_img):
    """
    Enhance the contrast of a grayscale image using histogram equalization.
    Input must be a grayscale image (2D array).
    """
    if gray_img.dtype != np.uint8:
        gray_img = gray_img.astype('uint8')
    return cv2.equalizeHist(gray_img)

# Apply color histogram equalization (LAB color space)
def apply_color_histogram_equalization(rgb_img):
    """
    Apply histogram equalization to color images using LAB color space.
    Equalizes the L channel and merges back with A and B channels.
    Returns RGB image.
    """
    # Ensure image is uint8
    if rgb_img.dtype != np.uint8:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    
    # Handle grayscale images
    if len(rgb_img.shape) == 2:
        return cv2.equalizeHist(rgb_img)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Equalize L channel
    l_eq = cv2.equalizeHist(l)
    
    # Merge back
    lab_eq = cv2.merge([l_eq, a, b])
    
    # Convert back to RGB
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return rgb_eq

# Show histogram visualization
def show_histogram(img, title="Image Histogram"):
    """
    Display histogram visualization for an image.
    Shows separate histograms for each RGB channel (R, G, B) and grayscale.
    """
    if len(img.shape) == 3:
        # Color image - show 4 subplots: R, G, B, and Grayscale
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # RGB channel names and colors
        channel_names = ['Red', 'Green', 'Blue']
        channel_colors = ['red', 'green', 'blue']
        
        # Calculate and plot each RGB channel separately
        for idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Extract channel and calculate histogram
            channel_data = img[:, :, idx]
            hist = cv2.calcHist([channel_data], [0], None, [256], [0, 256])
            ax.plot(hist, color=color, linewidth=2, label=name)
            ax.fill_between(range(256), hist.flatten(), alpha=0.3, color=color)
            ax.set_title(f'{name} Channel Histogram', fontsize=12, fontweight='bold')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Grayscale histogram in the 4th subplot
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
        axes[1, 1].plot(hist_gray, color='black', linewidth=2, label='Grayscale')
        axes[1, 1].fill_between(range(256), hist_gray.flatten(), alpha=0.3, color='gray')
        axes[1, 1].set_title('Grayscale Histogram', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        # Grayscale image - show single histogram
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(hist, color='black', linewidth=2, label='Grayscale')
        ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='gray')
        ax.set_title(f'{title} - Grayscale', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Load and preprocess all images in a folder
def preprocess_folder(folder_path):
    """
    Load, convert to grayscale, and equalize all images in a folder.
    Returns a list of processed image arrays.
    """
    processed_images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img_path = os.path.join(folder_path, filename)
            img = load_image(img_path)
            if img is not None:
                gray = to_gray(img)
                equalized = apply_histogram_equalization(gray)
                processed_images.append(equalized)
    return processed_images
