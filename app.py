import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from preprocessing import (
    load_image, 
    apply_color_histogram_equalization, 
    show_histogram
)
from modules.kmeans.kmeans_module import kmeans_image, visualize_clustered_pixels, visualize_cluster_colors
from classification import ImageClassifier

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("1400x900")
        
        # Set theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Variables
        self.original_image = None
        self.equalized_image = None
        self.clustered_image = None
        self.k_value = 5
        self.neighbors_value = 5
        
        # Initialize classifier with dataset if available
        dataset_path = os.path.join(project_root, "image_classification_project", "dataset")
        if os.path.exists(dataset_path):
            self.classifier = ImageClassifier(dataset_path=dataset_path, n_neighbors=self.neighbors_value)
        else:
            self.classifier = ImageClassifier(n_neighbors=self.neighbors_value)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header_frame = ctk.CTkFrame(self.root, fg_color="#FFFFFF", height=80)
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Icon and Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Image Classifier", 
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#235789"
        )
        title_label.pack(side="left", padx=30, pady=20)
        
        # Action Buttons
        btn_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=30, pady=20)
        
        self.upload_btn = ctk.CTkButton(
            btn_frame,
            text="📤 Upload Image",
            command=self.upload_image,
            width=140,
            height=36,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        self.upload_btn.pack(side="left", padx=5)
        
        self.save_btn = ctk.CTkButton(
            btn_frame,
            text="💾 Save Image",
            command=self.save_image,
            width=140,
            height=36,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        self.save_btn.pack(side="left", padx=5)
        
        self.reset_btn = ctk.CTkButton(
            btn_frame,
            text="↻ Reset",
            command=self.reset,
            width=100,
            height=36,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        self.reset_btn.pack(side="left", padx=5)
        
        # Main Content Area with scrolling
        scrollable_frame = ctk.CTkScrollableFrame(self.root, fg_color="#F5F5F5")
        scrollable_frame.pack(fill="both", expand=True, padx=30, pady=20)
        
        # Content wrapper frame for horizontal layout
        content_frame = ctk.CTkFrame(scrollable_frame, fg_color="#F5F5F5")
        content_frame.pack(fill="both", expand=True)
        
        # Left Panel - Images
        left_panel = ctk.CTkFrame(content_frame, fg_color="transparent")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Original Image
        orig_frame = ctk.CTkFrame(left_panel, fg_color="#FFFFFF", corner_radius=30)
        orig_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        orig_label = ctk.CTkLabel(
            orig_frame,
            text="🎨 Original Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#235789"
        )
        orig_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Canvas wrapper with rounded corners
        orig_canvas_frame = ctk.CTkFrame(orig_frame, fg_color="#E8E8E8", corner_radius=30)
        orig_canvas_frame.pack(padx=15, pady=(5, 2), fill="both", expand=True)
        
        self.orig_canvas = ctk.CTkCanvas(
            orig_canvas_frame,
            width=650,
            height=280,
            bg="#E8E8E8",
            highlightthickness=0
        )
        self.orig_canvas.pack(fill="both", expand=True)
        
        # Button frame to ensure consistent sizing
        orig_btn_frame = ctk.CTkFrame(orig_frame, fg_color="transparent", height=45)
        orig_btn_frame.pack(fill="x", padx=15, pady=(5, 15))
        orig_btn_frame.pack_propagate(False)
        
        self.show_orig_hist_btn = ctk.CTkButton(
            orig_btn_frame,
            text="📊 Show Original Histogram",
            command=self.show_original_histogram,
            width=280,
            height=40,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=16, weight="bold")
        )
        self.show_orig_hist_btn.pack(side="right", pady=2)
        
        # Equalized Image
        eq_frame = ctk.CTkFrame(left_panel, fg_color="#FFFFFF", corner_radius=30)
        eq_frame.pack(fill="both", expand=True)
        
        eq_label = ctk.CTkLabel(
            eq_frame,
            text="✨ Equalized Image",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#235789"
        )
        eq_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Canvas wrapper with rounded corners
        eq_canvas_frame = ctk.CTkFrame(eq_frame, fg_color="#E8E8E8", corner_radius=30)
        eq_canvas_frame.pack(padx=15, pady=(5, 2), fill="both", expand=True)
        
        self.eq_canvas = ctk.CTkCanvas(
            eq_canvas_frame,
            width=650,
            height=280,
            bg="#E8E8E8",
            highlightthickness=0
        )
        self.eq_canvas.pack(fill="both", expand=True)
        
        # Button frame to ensure consistent sizing
        eq_btn_frame = ctk.CTkFrame(eq_frame, fg_color="transparent", height=45)
        eq_btn_frame.pack(fill="x", padx=15, pady=(5, 15))
        eq_btn_frame.pack_propagate(False)
        
        self.show_eq_hist_btn = ctk.CTkButton(
            eq_btn_frame,
            text="📈 Show Equalized Histogram",
            command=self.show_equalized_histogram,
            width=280,
            height=40,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=16, weight="bold")
        )
        self.show_eq_hist_btn.pack(side="right", pady=2)
        
        # Right Panel - Controls
        right_panel = ctk.CTkFrame(content_frame, fg_color="transparent", width=320)
        right_panel.pack(side="right", fill="y", expand=False)
        right_panel.pack_propagate(False)
        
        # Preprocessing Section
        preprocess_frame = ctk.CTkFrame(right_panel, fg_color="#FFFFFF", corner_radius=30)
        preprocess_frame.pack(fill="x", pady=(0, 15))
        
        preprocess_header = ctk.CTkLabel(
            preprocess_frame,
            text="⚙️ PREPROCESSING",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#235789"
        )
        preprocess_header.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Image Equalization row with label and Apply button
        eq_row_frame = ctk.CTkFrame(preprocess_frame, fg_color="transparent")
        eq_row_frame.pack(fill="x", padx=15, pady=(10, 5))
        
        eq_label = ctk.CTkLabel(
            eq_row_frame,
            text="Image Equalization",
            font=ctk.CTkFont(size=13),
            text_color="#235789"
        )
        eq_label.pack(side="left")
        
        apply_btn = ctk.CTkButton(
            eq_row_frame,
            text="✓ Apply",
            command=self.apply_equalization,
            width=80,
            height=35,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        apply_btn.pack(side="right")
        
        # Auto-Resize checkbox
        self.auto_resize_var = ctk.BooleanVar(value=False)
        auto_resize_check = ctk.CTkCheckBox(
            preprocess_frame,
            text="Auto-Resize Image",
            variable=self.auto_resize_var,
            font=ctk.CTkFont(size=12),
            text_color="#235789"
        )
        auto_resize_check.pack(anchor="w", padx=15, pady=(5, 15))
        
        # K-Means Clustering Section
        kmeans_frame = ctk.CTkFrame(right_panel, fg_color="#FFFFFF", corner_radius=30)
        kmeans_frame.pack(fill="x", pady=(0, 15))
        
        kmeans_header = ctk.CTkLabel(
            kmeans_frame,
            text="🎛️ K-MEANS CLUSTERING",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#235789"
        )
        kmeans_header.pack(anchor="w", padx=15, pady=(15, 10))
        
        k_label = ctk.CTkLabel(
            kmeans_frame,
            text="K-value (clusters)",
            font=ctk.CTkFont(size=13),
            text_color="#235789"
        )
        k_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        k_value_frame = ctk.CTkFrame(kmeans_frame, fg_color="transparent")
        k_value_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        self.k_value_label = ctk.CTkLabel(
            k_value_frame,
            text="5",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#235789"
        )
        self.k_value_label.pack(side="right")
        
        self.k_slider = ctk.CTkSlider(
            kmeans_frame,
            from_=2,
            to=10,
            number_of_steps=8,
            command=self.update_k_value,
            width=260,
            button_color="#235789",
            button_hover_color="#1A4669",
            progress_color="#235789"
        )
        self.k_slider.set(5)
        self.k_slider.pack(padx=15, pady=(0, 10))
        
        slider_labels_frame = ctk.CTkFrame(kmeans_frame, fg_color="transparent")
        slider_labels_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(slider_labels_frame, text="2", font=ctk.CTkFont(size=10), text_color="#666666").pack(side="left")
        ctk.CTkLabel(slider_labels_frame, text="10", font=ctk.CTkFont(size=10), text_color="#666666").pack(side="right")
        
        self.run_clustering_btn = ctk.CTkButton(
            kmeans_frame,
            text="🚀 Run Clustering",
            command=self.run_clustering,
            width=260,
            height=36,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        self.run_clustering_btn.pack(padx=15, pady=(5, 15))
        
        # KNN Classification Section
        knn_frame = ctk.CTkFrame(right_panel, fg_color="#FFFFFF", corner_radius=30)
        knn_frame.pack(fill="x")
        
        knn_header = ctk.CTkLabel(
            knn_frame,
            text="🔍 KNN CLASSIFICATION",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#235789"
        )
        knn_header.pack(anchor="w", padx=15, pady=(15, 10))
        
        neighbors_label = ctk.CTkLabel(
            knn_frame,
            text="NEIGHBORS (K)",
            font=ctk.CTkFont(size=12),
            text_color="#666666"
        )
        neighbors_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.neighbors_dropdown = ctk.CTkComboBox(
            knn_frame,
            values=["3 Neighbors", "5 Neighbors", "7 Neighbors", "9 Neighbors"],
            width=260,
            height=36,
            corner_radius=30,
            command=self.update_neighbors
        )
        self.neighbors_dropdown.set("5 Neighbors")
        self.neighbors_dropdown.pack(padx=15, pady=(0, 10))
        
        self.classify_btn = ctk.CTkButton(
            knn_frame,
            text="🔍 Classify Image",
            command=self.classify_image,
            width=260,
            height=36,
            corner_radius=30,
            fg_color="#235789",
            hover_color="#1A4669",
            font=ctk.CTkFont(family="Inter", size=18, weight="bold")
        )
        self.classify_btn.pack(padx=15, pady=(5, 10))
        
        self.predicted_label = ctk.CTkLabel(
            knn_frame,
            text="Predicted class",
            font=ctk.CTkFont(size=12),
            text_color="#666666"
        )
        self.predicted_label.pack(pady=(5, 15))
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # Use preprocessing module to load image
            self.original_image = load_image(file_path)
            
            if self.original_image is not None:
                # Auto-resize is now applied during display, not during load
                self.display_image(self.original_image, self.orig_canvas)
                self.equalized_image = None
                self.eq_canvas.delete("all")
            else:
                messagebox.showerror("Error", "Failed to load image!")
    
    def display_image(self, img, canvas):
        # Resize image to fit canvas
        h, w = img.shape[:2]
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 650
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 280
        
        # If auto-resize is enabled, fill the entire frame
        if self.auto_resize_var.get():
            new_w = int(canvas_width * 0.95)
            new_h = int(canvas_height * 0.95)
        else:
            # Maintain aspect ratio
            scale = min(canvas_width/w, canvas_height/h) * 0.9
            new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor="center")
        canvas.image = img_tk
    
    def apply_equalization(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        
        try:
            # Use preprocessing module for color histogram equalization on full original image
            self.equalized_image = apply_color_histogram_equalization(self.original_image.copy())
            
            # Display the equalized image
            self.display_image(self.equalized_image, self.eq_canvas)
        except Exception as e:
            messagebox.showerror("Error", f"Equalization failed: {str(e)}")
    
    def get_working_image(self):
        """Get the image to work with based on auto-resize setting"""
        if self.auto_resize_var.get():
            return cv2.resize(self.original_image, (640, 480))
        return self.original_image
    
    def update_k_value(self, value):
        self.k_value = int(value)
        self.k_value_label.configure(text=str(self.k_value))
    
    def update_neighbors(self, choice):
        self.neighbors_value = int(choice.split()[0])
        # Update classifier neighbors
        self.classifier.set_neighbors(self.neighbors_value)
    
    def toggle_auto_resize(self):
        """Redisplay images when auto-resize is toggled"""
        if self.original_image is not None:
            self.display_image(self.original_image, self.orig_canvas)
        if self.equalized_image is not None:
            self.display_image(self.equalized_image, self.eq_canvas)
    
    def run_clustering(self):
        if self.equalized_image is None:
            messagebox.showwarning("No Image", "Please apply equalization first!")
            return
        
        try:
            # Use kmeans module for image clustering
            # Get both clustered image and labels for visualization
            self.clustered_image, cluster_labels = kmeans_image(
                self.equalized_image, 
                k=self.k_value, 
                max_iters=100,
                use_custom=True,  # Use custom kmeans implementation
                return_labels=True
            )
            
            self.display_image(self.clustered_image, self.eq_canvas)
            
            # Show visualization with pixels grouped by similar colors
            visualize_clustered_pixels(
                self.clustered_image, 
                cluster_labels,
                title=f"K-means Clustering Visualization (K={self.k_value})"
            )
            
            # Extract centroids for color visualization
            h, w = cluster_labels.shape
            unique_labels = np.unique(cluster_labels)
            centroids = []
            labels_flat = cluster_labels.flatten()
            
            for label in unique_labels:
                mask = (cluster_labels == label)
                cluster_pixels = self.clustered_image[mask]
                centroid = np.mean(cluster_pixels, axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Show cluster color visualization
            visualize_cluster_colors(
                centroids,
                labels_flat,
                title=f"Cluster Colors Analysis (K={self.k_value})"
            )
            
            messagebox.showinfo("Success", f"K-means clustering completed with {self.k_value} clusters!")
        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")
    
    def classify_image(self):
        if self.equalized_image is None:
            messagebox.showwarning("No Image", "Please process an image first!")
            return
        
        try:
            # Use classification module to predict
            predicted_class, confidence = self.classifier.predict(self.equalized_image)
            
            self.predicted_label.configure(
                text=f"Predicted class: {predicted_class}",
                text_color="#235789",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            
            if self.classifier.is_trained:
                available_classes = self.classifier.get_available_classes()
                class_list = ", ".join(available_classes)
                
                # Show warning if confidence is low
                confidence_msg = f"Confidence: {confidence:.2%}"
                if confidence < 0.4:
                    confidence_msg += " ⚠️ (Low confidence - prediction may be unreliable)"
                
                messagebox.showinfo(
                    "Classification", 
                    f"Image classified as: {predicted_class}\n{confidence_msg}\n\n"
                    f"Trained classes: {class_list}\n\n"
                    f"Note: The classifier can only predict classes it was trained on. "
                    f"If your image doesn't match any trained class, the prediction may be incorrect."
                )
            else:
                messagebox.showinfo(
                    "Classification", 
                    f"Image classified as: {predicted_class}\n\n(Note: Classifier not trained. Using random prediction. Train with dataset for accurate results.)"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def show_original_histogram(self):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please upload an image first!")
            return
        # Use full original image for histogram (not resized)
        # Use preprocessing module for histogram visualization
        show_histogram(self.original_image, "Original Image Histogram")
    
    def show_equalized_histogram(self):
        if self.equalized_image is None:
            messagebox.showwarning("No Image", "Please apply equalization first!")
            return
        # Use preprocessing module for histogram visualization
        show_histogram(self.equalized_image, "Equalized Image Histogram")
    
    def save_image(self):
        # Save clustered image if available, otherwise save equalized image
        image_to_save = None
        if self.clustered_image is not None:
            image_to_save = self.clustered_image
        elif self.equalized_image is not None:
            image_to_save = self.equalized_image
        else:
            messagebox.showwarning("No Image", "No processed image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        if file_path:
            try:
                img_bgr = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def reset(self):
        self.original_image = None
        self.equalized_image = None
        self.clustered_image = None
        self.orig_canvas.delete("all")
        self.eq_canvas.delete("all")
        self.k_slider.set(5)
        self.k_value = 5
        self.k_value_label.configure(text="5")
        self.neighbors_dropdown.set("5 Neighbors")
        self.predicted_label.configure(text="Predicted class", text_color="#666666", font=ctk.CTkFont(size=12))


if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageClassifierApp(root)
    root.mainloop()
