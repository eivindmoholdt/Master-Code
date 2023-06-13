"""
DYNAMIC IMAGE GENERERATION MODEL
This script automatically runs image generation and prediciton from start to finish for 5 samples per model:
Text processing, Image Generation + JSON files, Object Detection + encoder, Similarity Checker and Predictions
Change number of captions to read from test_data.json file in the Generation scripts in order 
to change number of generated images.
This script servers as a easy, ready-to-go demonstration and a dynamic model for testing our approach
"""

from image_generation.DALLE.DALLE2gen import DALLE2_gen
from image_generation.SD.SDgen import SDgen
from DALLEjson import dallejson
from SDjson import SDjson
from config import *
import config
from ODOEcalcsim import calc_sim_yolo, calc_sim_maskrcnn
from OEcalcsim import calc_sim_encoder
import json
import statistics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


#Image Generation
if config.DALLE2:
    DALLE2_gen()
    dallejson()
    jsonfile = "DALLE.json"
if config.SD:
    SDgen()
    SDjson()
    jsonfile = "SDv14.json"

#Object Detection and Object Encoder
if config.USE_DETECTION:
    if config.YOLOV5 or config.YOLOV7:
        similarity = calc_sim_yolo()
    if config.MASKRCNN:
        similarity = calc_sim_maskrcnn()
else:
    #Only Object Encoder
    similarity = calc_sim_encoder()
    
#Predictions

#you can adjust threshold in config. In order to get an even prediction of OOC/NOOC values we recommend using statistics.median of sim_scores as threshold
gold_labels = []
predicted_labels = []
sim_scores = []
with open(jsonfile) as f:
    my_dict = [json.loads(line) for line in f]

    mappe = []
    for i in my_dict:
        img1 = i['img_gen1']
        img2 = i['img_gen2']
        label = i['label']
        path = i['img_local_path']
        gold_labels.append(label)
        mappe.append([img1, img2])

    for i in range(len(mappe)):
        sim = similarity(mappe[i])
        sim_scores.append(sim)

    #Remove threshold if you want to set threshold in config
    threshold = statistics.median(sim_scores)
    for x in sim_scores:
        if sim < threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

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



