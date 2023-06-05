import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import spacy
import os
import pandas as pd
import torch
from torchtext import data
import json
import cv2
spacy_en = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language="english")
from torchtext.vocab import GloVe, FastText
#We use the utils from COSMOS to load the test.json file. CD to COSMOS folder or add COSMOS.utils.config and COSMOS.utils.common_utils to get right path
from COSMOS.utils.config import DATA_DIR
from COSMOS.utils.common_utils import read_json_data
from nltk.stem.snowball import SnowballStemmer



""" Pretrained Stable Diffusion model from HuggingFace.
The dataset generated with this model is saved to SDv14-GeneratedDataset
 "CompVis/stable-diffusion-v1-4", revision="fp16",torch_dtype=torch.float16, use_auth_token=True"

* Note from documentation: If you are limited by GPU memory and have less than 4GB of GPU RAM available, 
please make sure to load the StableDiffusionPipeline in float16 precision instead of the default float32 precision. 
You can do so by telling diffusers to expect the weights to be in float16 precision. *

Running the model with float32 in Google Colab (normal GPU version) will crash the runtime

 See https://huggingface.co/CompVis/stable-diffusion-v1-4 for info on the model.

"""


from huggingface_hub import notebook_login
notebook_login()
#add valid token from https://huggingface.co/settings/tokens to be able to run the Stable Diffusion model


#Load the Stable Diffusion Pipeline.
experimental_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True) 
#Add pipe.to("cuda") to utilize GPU
experimental_pipe = experimental_pipe.to("cuda")


# Generate specific images
test_data = read_json_data(os.path.join(DATA_DIR, 'test_data.json')) [0:5]
#Generate all images
#test_data = read_json_data(os.path.join(DATA_DIR, 'test_data.json'))

for i in test_data:
  description_1 = i['caption1_modified']
  description_2 = i['caption2_modified']
  with autocast("cuda"):
    image_1 = experimental_pipe(description_1).images[0]
    image_2 = experimental_pipe(description_2).images[0]

    print(i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'}))
  
  #saves images to 'filename_gen1.jpg', translate function to remove .jpg and .png extension for original image
  # so we are able to map the generated images back to the original image
  image_1.save(f"SDv14-GeneratedDataset/new/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}gen1.jpg")
  image_2.save(f"SDv14-GeneratedDataset/new/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}gen2.jpg")

"""
time: 3h 38min 34s 

3 seconds per image
"""

