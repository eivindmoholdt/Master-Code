import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the pre-trained CNN model
model = VGG16(weights='imagenet', include_top=False)

# Define a function to extract features from an image using the CNN model
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    features = model.predict(np.array([img]))
    return features.flatten()

# Load the input images
img1_path = 'C:\Users\IL4559\Pictures\DALLE\1.png'
img2_path = 'C:\Users\IL4559\Pictures\DALLE\2.png'

# Extract features from the input images
features1 = extract_features(img1_path)
features2 = extract_features(img2_path)

# Calculate the distance between the feature vectors of the two images
distance = np.linalg.norm(features1 - features2)

# Print the similarity score
print('Similarity score:', 1 / (1 + distance))
