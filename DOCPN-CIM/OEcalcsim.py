import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from config import *
from sklearn.metrics.pairwise import cosine_similarity


def calc_sim_encoder(image_pairs):
    for encoder_name, encoder in encoders.items():
        print(f"Using {encoder_name} Encoder\n{'-'*30}")
   
        # Set the encoder to evaluation mode
        encoder.eval()

        feature_vectors = []
        for image_path in image_pairs:
            img = cv2.imread(image_path)
        
            # Define image transforms
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            with torch.no_grad():
                img = Image.open(image_path) # Convert to PIL Image
                image_tensor = transform(img).unsqueeze(0)
                encoder_features = encoder(image_tensor).squeeze().numpy()
            feature_vectors.append(encoder_features)

        a = feature_vectors[0]
        b = feature_vectors[1]

        # Reshape the arrays to be 2D arrays with one row
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

        # Compute the cosine similarity between the arrays
        similarity = cosine_similarity(a, b)

    return similarity[0][0]
