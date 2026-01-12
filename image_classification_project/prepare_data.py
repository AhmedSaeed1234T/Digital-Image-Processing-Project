import os
import cv2
from sklearn.model_selection import train_test_split
dataset_path = "dataset/"

images = []
labels = []

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        images.append(img)
        labels.append(folder)

unique_labels = list(set(labels))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}

numeric_labels = [label_to_num[l] for l in labels]

normalized_images = []

for img in images:
    resized = cv2.resize(img, (128, 128))
    normalized_images.append(resized)

X_train, X_test, y_train, y_test = train_test_split(
    normalized_images,
    numeric_labels,
    test_size=0.2,
    random_state=42
)

print("Total images:", len(images))
print("Train:", len(X_train))
print("Test:", len(X_test))
print("Label mapping:", label_to_num)
