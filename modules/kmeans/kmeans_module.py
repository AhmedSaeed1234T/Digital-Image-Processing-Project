import numpy as np
import matplotlib.pyplot as plt
import cv2

# =============================================
# 1) Initialize Centroids
# =============================================
def initialize_centroids(data, k):
    """
    Randomly select k unique data points as initial centroids
    """
    np.random.seed(42)
    random_indices = np.random.choice(len(data), size=k, replace=False)
    return data[random_indices]


# =============================================
# 2) Assign Each Point to the Nearest Centroid
# =============================================
def assign_clusters(data, centroids):
    """
    Calculate Euclidean distance between each point and each centroid
    Return an array of cluster labels
    """
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


# =============================================
# 3) Update Centroids
# =============================================
def update_centroids(data, labels, k):
    """
    Recalculate the centroid of each cluster as a mean of assigned points
    """
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


# =============================================
# 4) K-means Main Loop
# =============================================
def kmeans(data, k, max_iters=100):
    """
    Full K-means algorithm implementation
    """
    centroids = initialize_centroids(data, k)

    for i in range(max_iters):
        old_centroids = centroids.copy()

        # Step 1: Assign clusters
        labels = assign_clusters(data, centroids)

        # Step 2: Update centroids
        centroids = update_centroids(data, labels, k)

        # Check convergence (if centroids did not change)
        if np.allclose(centroids, old_centroids):
            print(f"Converged after {i+1} iterations.")
            break

    return centroids, labels


# =============================================
# 5) Plot Clusters
# =============================================
def plot_clusters(data, labels, centroids, title="K-means Clustering"):
    """
    Visualize clusters and centroids (2D only)
    """
    plt.figure(figsize=(8, 6))

    k = len(centroids)
    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                marker='*', s=300, c='black', label="Centroids")

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================
# 6) Apply K-means to Images
# =============================================
def kmeans_image(image, k, max_iters=100, use_custom=True, return_labels=False):
    """
    Apply K-means clustering to an image.
    
    Args:
        image: Input image (RGB format, numpy array)
        k: Number of clusters
        max_iters: Maximum iterations
        use_custom: If True, use custom kmeans implementation; else use sklearn
        return_labels: If True, also return cluster labels
    
    Returns:
        clustered_image: Image with pixels replaced by cluster centers
        labels (optional): Cluster labels for each pixel
    """
    # Get image shape
    h, w, c = image.shape
    
    # Reshape image to 2D array (pixels x channels)
    img_reshaped = image.reshape(-1, c)
    
    # Convert to float32
    img_float = img_reshaped.astype(np.float32)
    
    if use_custom:
        # Use custom kmeans implementation
        centroids, labels = kmeans(img_float, k, max_iters)
        
        # Map labels to cluster centers
        clustered_pixels = centroids[labels]
    else:
        # Use sklearn KMeans
        from sklearn.cluster import KMeans
        kmeans_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=max_iters)
        labels = kmeans_sklearn.fit_predict(img_float)
        centroids = kmeans_sklearn.cluster_centers_
        clustered_pixels = centroids[labels]
    
    # Convert back to uint8 and reshape to original image shape
    clustered_image = np.clip(clustered_pixels, 0, 255).astype(np.uint8).reshape(h, w, c)
    
    if return_labels:
        return clustered_image, labels.reshape(h, w)
    return clustered_image


def visualize_clustered_pixels(image, labels, title="Clustered Pixels Visualization"):
    """
    Visualize clustered pixels by grouping similar colors together.
    Shows pixels rearranged so that pixels in the same cluster are next to each other.
    
    Args:
        image: Original image (RGB format)
        labels: Cluster labels for each pixel (2D array, same shape as image height/width)
        title: Title for the visualization
    """
    h, w = labels.shape
    c = image.shape[2] if len(image.shape) == 3 else 1
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)
    
    # Create figure with 2 subplots: original clustered image and rearranged visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Left plot: Show the clustered image (original layout)
    axes[0].imshow(image)
    axes[0].set_title('Clustered Image (Original Layout)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Right plot: Rearrange pixels by cluster
    # Create a new image where pixels are sorted by cluster
    rearranged_image = np.zeros_like(image)
    
    # For each cluster, collect all pixels belonging to it
    cluster_pixels = {}
    cluster_positions = {}
    
    for cluster_id in unique_clusters:
        mask = (labels == cluster_id)
        cluster_pixels[cluster_id] = image[mask]
        cluster_positions[cluster_id] = np.where(mask)
    
    # Rearrange: place pixels of same cluster together
    # Strategy: Create blocks for each cluster
    current_y = 0
    for cluster_id in unique_clusters:
        pixels = cluster_pixels[cluster_id]
        num_pixels = len(pixels)
        
        # Calculate how many rows we need for this cluster
        rows_needed = max(1, int(np.ceil(num_pixels / w)))
        
        # Place pixels row by row
        pixel_idx = 0
        for row in range(rows_needed):
            if current_y + row >= h:
                break
            for col in range(w):
                if pixel_idx < num_pixels:
                    rearranged_image[current_y + row, col] = pixels[pixel_idx]
                    pixel_idx += 1
                else:
                    break
            if pixel_idx >= num_pixels:
                break
        
        current_y += rows_needed
        if current_y >= h:
            break
    
    # Show rearranged image
    axes[1].imshow(rearranged_image)
    axes[1].set_title('Pixels Grouped by Similar Colors (Same Cluster Together)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_cluster_colors(centroids, labels, title="Cluster Color Visualization"):
    """
    Visualize the dominant colors found by k-means clustering.
    
    Args:
        centroids: Cluster centroids (k x 3 array for RGB)
        labels: Cluster labels
        title: Title for visualization
    """
    k = len(centroids)
    
    # Create a color palette showing each cluster's dominant color
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Left: Color swatches for each cluster
    color_swatches = np.zeros((100, k * 100, 3), dtype=np.uint8)
    for i, centroid in enumerate(centroids):
        color = np.clip(centroid, 0, 255).astype(np.uint8)
        color_swatches[:, i*100:(i+1)*100] = color
    
    axes[0].imshow(color_swatches)
    axes[0].set_title('Dominant Colors (Cluster Centroids)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Bar chart showing pixel count per cluster
    cluster_counts = [np.sum(labels == i) for i in range(k)]
    colors_rgb = [np.clip(centroids[i], 0, 255).astype(np.uint8) / 255.0 for i in range(k)]
    
    bars = axes[1].bar(range(k), cluster_counts, color=colors_rgb)
    axes[1].set_xlabel('Cluster ID', fontsize=11)
    axes[1].set_ylabel('Number of Pixels', fontsize=11)
    axes[1].set_title('Pixel Distribution Across Clusters', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(k))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, cluster_counts)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/np.sum(cluster_counts)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# =============================================
# 7) Test Code (Optional)
# =============================================
if __name__ == "__main__":
    # Example test on random data
    from sklearn.datasets import make_blobs
    
    data, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    k = 3
    centroids, labels = kmeans(data, k)
    
    plot_clusters(data, labels, centroids)
