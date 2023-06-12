"""
DYNAMIC IMAGE GENERERATION MODEL
This script automatically runs image generation and prediciton from start to finish for 5 samples per model:
Text processing, Image Generation, Object Detection + encoder, Similarity Checker and Predictions
Change number of captions to read from test_data.json file in the Generation scripts in order 
to change number of generated images.
This script servers as a easy, ready-to-go demonstration and a dynamic model for testing our approach
"""

from image_generation.DALLE import *
from image_generation.SD import *
from config import *
import config
from ODOEcalcsim import calc_sim_yolo, calc_sim_maskrcnn, set_detection_model
from OEcalcsim import calc_sim_encoder

#Image Generation


#Object Detection and Object Encoder
if config.USE_DETECTION:
    if config.YOLOV5 or config.YOLOV7:
        calc_sim_yolo()
    if config.MASKRCNN:
        calc_sim_maskrcnn()
else:
    calc_sim_encoder()
        
#Only Object Encoder


#Similarity checker

#Predictions

#Compare with predictions from COSMOS?

#Results



