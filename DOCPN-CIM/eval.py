import cv2
import numpy as np
import torch
from google.colab.patches import cv2_imshow
import json
from yolov5resnet_similarity import calc_sim

#Evaluation with YOLOv5 and ResNet

# First, we run object Detection using a pre-trained YOLOv5 model to get bounding boxes
# and crop the bounding boxes as images, to feed into ResNET for feature extraction


# Load pre-trained ResNet50 model
resnet50 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

# Set the model to evaluation mode
resnet50.eval()

threshold = 0.5
gold_labels = []
predicted_labels = []


with open("SDv14.json") as f:
  my_dict = [json.loads(line) for line in f]

  mappe = []
  for i in my_dict:
    img1 = i['img_gen1']
    img2 = i['img_gen2']
    label = i['label']
    gold_labels.append(label)
    mappe.append([img1, img2])

  for i in range(len(mappe)):
      sim = calc_sim(mappe[i])
      if sim < threshold:
        predicted_labels.append(1)
      else:
        predicted_labels.append(0)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

confusion_mat = confusion_matrix(gold_labels, predicted_labels)
accuracy = accuracy_score(gold_labels, predicted_labels)
precision = precision_score(gold_labels, predicted_labels)
recall = recall_score(gold_labels, predicted_labels)
f1 = f1_score(gold_labels, predicted_labels)

print("Confusion Matrix:\n", confusion_mat)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)