import os
import cv2
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from models import extract_color, extract_edge
except ImportError:
    # Fallback if import fails
    import sys
    import os
    # Try adding current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(current_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from models import extract_color, extract_edge

class ImageClassifier:
    def __init__(self, dataset_path=None, n_neighbors=2):
        """
        Initialize the image classifier.
        
        Args:
            dataset_path: Path to the dataset folder containing class subfolders
            n_neighbors: Number of neighbors for KNN (default: 5)
        """
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.label_to_num = {}
        self.num_to_label = {}
        self.feature_type = 'combined'  # 'color', 'edge', or 'combined'
        self.is_trained = False
        self.scaler = None  # For feature normalization
        
        if dataset_path:
            self.load_and_train(dataset_path)
    
    def load_dataset(self, dataset_path):
        """
        Load images and labels from dataset folder.
        Each subfolder represents a class.
        
        Returns:
            images: List of image arrays
            labels: List of class labels (folder names)
        """
        images = []
        labels = []
        
        if not os.path.exists(dataset_path):
            return images, labels
        
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            
            if not os.path.isdir(folder_path):
                continue
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(folder)
        
        return images, labels
    
    def extract_features(self, images, feature_type='color'):
        """
        Extract features from a list of images.
        
        Args:
            images: List of image arrays
            feature_type: 'color' for color histogram, 'edge' for edge features
        
        Returns:
            features: Array of feature vectors
        """
        features = []
        for img in images:
            if feature_type == 'color':
                feat = extract_color(img)
            elif feature_type == 'edge':
                feat = extract_edge(img)
            else:
                # Combine both features
                feat_color = extract_color(img)
                feat_edge = extract_edge(img)
                feat = np.concatenate([feat_color, feat_edge])
            features.append(feat)
        return np.array(features)
    
    def load_and_train(self, dataset_path, feature_type='combined', test_size=0.2):
        """
        Load dataset, extract features, and train the classifier.
        
        Args:
            dataset_path: Path to dataset folder
            feature_type: 'color', 'edge', or 'combined' (default: 'combined' for better accuracy)
            test_size: Proportion of data to use for testing
        """
        print("Loading dataset...")
        images, labels = self.load_dataset(dataset_path)
        
        if len(images) == 0:
            print("Warning: No images found in dataset. Classifier will use random predictions.")
            return
        
        # Create label mappings
        unique_labels = list(set(labels))
        self.label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_to_label = {idx: label for label, idx in self.label_to_num.items()}
        
        print(f"Found {len(images)} images in {len(unique_labels)} classes: {unique_labels}")
        
        # Show class distribution
        from collections import Counter
        label_counts = Counter(labels)
        print("Class distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} images")
        
        # Extract features
        print(f"Extracting {feature_type} features...")
        self.feature_type = feature_type
        features = self.extract_features(images, feature_type)
        
        print(f"Feature vector size: {features.shape[1]} features")
        
        # Normalize features for better classification
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        # Convert labels to numeric
        numeric_labels = [self.label_to_num[label] for label in labels]
        
        # Split data
        # If we have very few samples, don't split (use all for training)
        if test_size > 0 and len(features) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                features, numeric_labels, test_size=test_size, random_state=42
            )
        else:
            X_train, y_train = features, numeric_labels
            X_test, y_test = None, None
            if test_size > 0 and len(features) <= 5:
                print(f"Warning: Only {len(features)} samples available. Using all for training (no test split).")
        
        # Adjust n_neighbors if it's greater than number of training samples
        max_neighbors = min(self.n_neighbors, len(X_train))
        if max_neighbors < self.n_neighbors:
            print(f"Warning: Adjusting n_neighbors from {self.n_neighbors} to {max_neighbors} (only {len(X_train)} training samples available)")
            self.n_neighbors = max_neighbors
        
        # Train classifier with distance weighting for better accuracy
        print(f"Training KNN classifier with {self.n_neighbors} neighbors...")
        # Use distance weighting so closer neighbors have more influence
        # Use 'distance' weighting instead of uniform
        self.classifier = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights='distance',  # Weight by inverse distance
            metric='euclidean',
            algorithm='auto'
        )
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        if X_test is not None and len(X_test) > 0:
            try:
                accuracy = self.classifier.score(X_test, y_test)
                print(f"Training complete! Test accuracy: {accuracy:.2%}")
            except ValueError as e:
                print(f"Training complete! (Could not compute test accuracy: {str(e)})")
        else:
            print("Training complete!")
    
    def predict(self, image):
        """
        Predict the class of a single image.
        
        Args:
            image: Image array (RGB format)
        
        Returns:
            predicted_class: Predicted class name
            confidence: Confidence score (if available)
        """
        if not self.is_trained:
            # Return random prediction if not trained
            if len(self.label_to_num) > 0:
                classes = list(self.label_to_num.keys())
                return np.random.choice(classes), 0.0
            else:
                classes = ["Cat", "Dog", "Bird", "Car", "Flower"]
                return np.random.choice(classes), 0.0
        
        # Extract features
        features = self.extract_features([image], self.feature_type)
        
        # Normalize features using the same scaler used during training
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Predict
        prediction_num = self.classifier.predict(features)[0]
        predicted_class = self.num_to_label[prediction_num]
        
        # Get probabilities if available
        try:
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = probabilities[prediction_num]
            
            # Check if confidence is too low (might be unreliable)
            max_prob = np.max(probabilities)
            if max_prob < 0.4:  # If max probability is less than 40%, prediction is uncertain
                # Return the prediction but with low confidence
                confidence = max_prob
        except:
            confidence = 1.0
        
        return predicted_class, confidence
    
    def get_available_classes(self):
        """
        Get list of classes the classifier was trained on.
        """
        return list(self.label_to_num.keys())
    
    def set_neighbors(self, n_neighbors):
        """
        Update the number of neighbors and retrain if classifier exists.
        """
        if self.classifier is not None and self.is_trained:
            # Get the number of training samples
            n_samples = self.classifier.n_samples_fit_
            # Adjust n_neighbors if needed
            max_neighbors = min(n_neighbors, n_samples)
            if max_neighbors < n_neighbors:
                # Silently adjust to avoid UI issues
                self.n_neighbors = max_neighbors
            else:
                self.n_neighbors = n_neighbors
            
            # Update classifier with adjusted n_neighbors
            self.classifier.n_neighbors = self.n_neighbors
        else:
            self.n_neighbors = n_neighbors

