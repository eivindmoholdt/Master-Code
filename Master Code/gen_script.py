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

#Text Processing Step
if text_processing_NER == True:
    from utils.text_utils import modify_caption_replace_entities
    modify_caption_replace_entities(caption)
else:
    pass

#Censor captions
if censoring == True:
    pass
else:
    pass

#model choice
if DALLE==True:
    pass
else:
    pass

if SD==True:
    pass
else:
    pass


#Object Detection


#Object Encoder

#Similarity checker

#Predictions

#Compare with predictions from COSMOS?

#Results



