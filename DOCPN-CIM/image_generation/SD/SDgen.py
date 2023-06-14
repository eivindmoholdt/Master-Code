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
from pathlib import Path
from torchtext.vocab import GloVe, FastText
#We use the utils from COSMOS to load the test.json file. CD to COSMOS folder or add COSMOS.utils.config and COSMOS.utils.common_utils to get right path
from config import COSMOS_DATA, HFTOKEN
nlp = spacy.load("en_core_web_sm")


#Function borrowed from COSMOS code at COSMOS.utils.dataset_utils.py
def modify_caption_replace_entities(caption_text):
    """
        Utility function to replace named entities in the caption with their corresponding hypernyms

        Args:
            caption_text (str): Original caption with named entities

        Returns:
            caption_modified (str): Modified caption after replacing named entities
    """
    doc = nlp(caption_text)
    caption_modified = caption_text
    caption_entity_list = []
    for ent in doc.ents:
        caption_entity_list.append((ent.text, ent.label_))
        caption_modified = caption_modified.replace(ent.text, ent.label_, 1)
    return caption_modified


""" Pretrained Stable Diffusion model from HuggingFace.
The dataset generated with this model is saved to SDv14-GeneratedDataset
 "CompVis/stable-diffusion-v1-4", revision="fp16",torch_dtype=torch.float16, use_auth_token=True"

* Note from documentation: If you are limited by GPU memory and have less than 4GB of GPU RAM available, 
please make sure to load the StableDiffusionPipeline in float16 precision instead of the default float32 precision. 
You can do so by telling diffusers to expect the weights to be in float16 precision. *

Running the model with float32 in Google Colab (normal GPU version) will crash the runtime

Note: Pipelines loaded with `torch_dtype=torch.float16` cannot run with `cpu` device. It is not recommended to move them to `cpu` as running them will fail. 

 See https://huggingface.co/CompVis/stable-diffusion-v1-4 for info on the model.

"""



#Load the Stable Diffusion Pipeline.
experimental_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=HFTOKEN) 
#Add pipe.to("cuda") to utilize GPU
experimental_pipe = experimental_pipe.to("cuda")

def SDgen():
  IMAGE_DIR = Path.cwd() / "NewDatasets/SD"
  IMAGE_DIR.mkdir(parents=True, exist_ok=True)
  # Generate specific images
  with open(os.path.join(COSMOS_DATA, 'test_data.json')) as f:
    test_data = [json.loads(line) for line in f][0:5]

  for i in test_data:
    description_1 = i['caption1']
    description_2 = i['caption2']

    #NER tagging. We use the same code and model from COSMOS to ensure comparability.
    # You can also just load the caption1_modified and caption2_modified
    description_1 = modify_caption_replace_entities(description_1)
    description_2 = modify_caption_replace_entities(description_2)

    #Censoring
    #We split words on comma delimiter. This way we can add conconated words to the list
    with open('badwords.txt','r') as f:
      for line in f:
        for word in line.split(","):
          description_1 = description_1.replace(word, ''*len(word))
          description_2 = description_2.replace(word, ''*len(word))

    
    with autocast("cuda"):
      image_1 = experimental_pipe(description_1).images[0]
      image_2 = experimental_pipe(description_2).images[0]

      print(i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'}))
    
    #saves images to 'filename_gen1.jpg', translate function to remove .jpg and .png extension for original image
    # so we are able to map the generated images back to the original image
    image_1.save(f"NewDatasets/SD/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}gen1.jpg")
    image_2.save(f"NewDatasets/SD/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}gen2.jpg")

"""
time: 3h 38min 34s 

3 seconds per image on Premium Colab GPU
15 seconds per image on normal Colab GPU
"""

