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

#Results



