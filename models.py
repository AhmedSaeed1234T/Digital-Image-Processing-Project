import numpy as np
import cv2

def extract_color(image):
    """
    Extract improved color histogram features from an image.
    Uses normalized histograms and statistical features for better discrimination.
    """
    # Ensure image is in correct format
    if image is None:
        return np.zeros(780)  # 768 (histogram) + 12 (statistical features)
    
    # Resize for consistent feature size
    image = cv2.resize(image, (128, 128))
    
    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    feature = []
    if len(image.shape) == 3 and image.shape[-1] == 3:
        # RGB image - extract normalized histogram for each channel
        for i in range(3):
            channel = image[:, :, i]
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            # Normalize histogram to sum to 1 (probability distribution)
            hist = hist / (hist.sum() + 1e-6)  # Add small epsilon to avoid division by zero
            feature.extend(hist.flatten())
            
            # Add statistical features for each channel
            feature.append(channel.mean() / 255.0)  # Normalized mean
            feature.append(channel.std() / 255.0)  # Normalized std
            feature.append(np.percentile(channel, 25) / 255.0)  # Q1
            feature.append(np.percentile(channel, 75) / 255.0)  # Q3
    else:
        # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-6)
        feature = list(hist.flatten())
        # Add statistical features
        feature.append(image.mean() / 255.0)
        feature.append(image.std() / 255.0)
        feature.append(np.percentile(image, 25) / 255.0)
        feature.append(np.percentile(image, 75) / 255.0)
        # Pad to same size as RGB
        feature = feature * 3
    
    return np.array(feature, dtype=np.float32)


def extract_edge(image):
    """
    Extract improved edge features using Canny edge detection and texture analysis.
    Returns statistical features from edges rather than raw pixels.
    """
    if image is None:
        return np.zeros(20)  # Statistical features instead of raw pixels
    
    # Resize for consistent feature size
    image = cv2.resize(image, (128, 128))
    
    # Ensure uint8 format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Canny edge detection with multiple thresholds
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 100, 200)
    
    # Extract statistical features from edges
    features = []
    
    # Edge density (percentage of edge pixels)
    features.append(np.sum(edges1 > 0) / (edges1.size + 1e-6))
    features.append(np.sum(edges2 > 0) / (edges2.size + 1e-6))
    
    # Edge strength statistics
    features.append(edges1.mean() / 255.0)
    features.append(edges1.std() / 255.0)
    features.append(edges2.mean() / 255.0)
    features.append(edges2.std() / 255.0)
    
    # Texture features using gradient
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.append(gradient_magnitude.mean() / 255.0)
    features.append(gradient_magnitude.std() / 255.0)
    features.append(np.percentile(gradient_magnitude, 50) / 255.0)
    features.append(np.percentile(gradient_magnitude, 90) / 255.0)
    
    # Local Binary Pattern (LBP) texture (simplified)
    # Calculate local variance as texture measure
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    features.append(local_var.mean() / (255.0**2))
    features.append(local_var.std() / (255.0**2))
    
    # Histogram of Oriented Gradients (HOG) - simplified version
    # Calculate gradient orientations
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angles = angles[angles < 0] + 180  # Normalize to 0-180
    
    # Histogram of orientations (8 bins)
    hist, _ = np.histogram(angles, bins=8, range=(0, 180))
    hist = hist / (hist.sum() + 1e-6)  # Normalize
    features.extend(hist.tolist())
    
    return np.array(features, dtype=np.float32)
