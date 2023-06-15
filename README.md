# Detecting Out-of-Context Image-Caption Pairs in News: A Counter-Intuitive Method

![Teaser Image](https://github.com/eivindmoholdt/Master-Code/blob/main/DOCPN-CIM/Teaser/finalfinalfinalmodelarch.jpg?raw=True)

The core idea of this method is to utilize the perceptual similarity between synthetically generated images (from both DALLE-2 and Stable Diffusion) for detecting Out-Of-Context content in image and caption pairs. By comparing the original image to the generated images, we successfully detect Out-Of-Context (OOC) in the left image-caption triplets, and Not-Out-of-Context (NOOC) in the right image-caption triplets. The original captions 1 and 2 are presented below each correlating generated image.

### Datasets
Unzip the dataset zips in Datasets folder into corresponding folders for DALL-E 2 and SD.

#### Important!
The COSMOS dataset is not public. If you want to test our approach, visit https://detecting-cheapfakes.github.io/ or fill out https://docs.google.com/forms/d/e/1FAIpQLSf7rZ1-UX419nXqCp2NldekqVNJcS2W9A3jL7MTKhom41p0eg/viewform to get access.
See the COSMOS github for more information.

In order to map the generated images to the original images and the captions, we create a JSON file that contains the mapping between the generated images, original and modified captions, original image path and labels.
After downloading the COSMOS dataset, run the scripts DALLEjson.py and SDjson.py in order to generate the JSON files to map the generated images.
The JSON files will be in the structure as below:

```json
{
  "img_local_path": "<img_path>",
  "original_caption1": "<caption1>",
  "caption1_mod": "<caption1_modified>",
  "img_gen1": "<generated_image1>",
  "original_caption2": "<caption2>",
  "caption2_mod": "<caption2_modified>",
  "img_gen2": "<generated_image2>",
  "label": "ooc/not-ooc"
}
```

* img_local_path: The original image file path name from the COSMOS test dataset.
* original_caption1: The original caption1 associated with the original image from the
COSMOS test dataset
* caption1_mod: The modified caption1, post NER and bad word censoring, associated
with the original image.
* img_gen1: The generated image associated to (modified) caption1.
* original_caption2: The original caption2 associated with the original image from the
COSMOS test dataset
* caption2_mod: The modified caption2, post NER and bad word censoring, associated
with the original image.
* img_gen2: The generated image associated to (modified) caption2.
* label: Class label whether the two captions are out-of-context with respect to the image (1=Out-of-Context, 0=Not-Out-of-Context ), from the COSMOS test dataset.



### Requirements and pre-requisites:
All the code in this project was written and run in Google Colab. 
We recommend running all the code in this project in Google Colab or similar.
The Python files has been written to allow easy automatic testing of our model. 
To test parts of the code we have included the raw IPYNB files in '/Google' folder.

Install the packages from requirements.txt using:
pip install -r 'requirements.txt'
Note: This also includes necesarry packages for the COSMOS model. However, package versions might not be the same. If you want to run the COSMOS model, see the COSMOS folder requirements.txt. and readme files.

DALL-E 2 (Open AI API) and Stable Diffusion (HuggingFace) setups are detailed in the respective python files in '/image_generation'

##### CLIP:
CLIP can be loaded by Git cloning git+https://github.com/openai/CLIP.git or via SentenceTransformers library
SentenceTransformers is included in requirements.txt.
SentenceTransformers library built on Pytorch allows for the easiest implementation:
model = SentenceTransformer('clip-ViT-L-14') or 
model = SentenceTransformer('clip-ViT-B-32')

See https://huggingface.co/sentence-transformers/clip-ViT-B-32
and https://huggingface.co/sentence-transformers/clip-ViT-L-14
for documentation

##### YOLOv5:
YOLOv5 is easily implemented with Pytorch
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained=True)
You can set pretrained=True or instead download the weights for the model and set path='path_to_weights'
Parameter 'yolov5' can be changed to 'yolov5s' or similar if you want specific model versions
See https://pytorch.org/hub/ultralytics_yolov5/ for more info

##### YOLOv7:
Clone YOLOv7 github rep with ! git clone https://github.com/WongKinYiu/yolov7.git
Import the custom function with:
from yolov7.hubconf import custom
yolov7 = custom('yolov7.pt') #add correct path to .pt file
after this YOLOv7 can be used the same way as YOLOv5

##### MASK-RCNN:
Loaded trough Detectron2 with !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
See Detectron2 docs for guide:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

## Image Generation
The Image Generation folder contains the Python files DALL-E-2_Gen.py and SD_Gen.py, for generating datasets with DALL-E 2 and SD.
We load the captions from the original test_data.json file from COSMOS for image generation.

CardinalImages folder contains the filepaths to images from DALL-E 2 and SD clearly affected by the 'Cardinal' entity label.

## Predictions and evaluation
The main.py file servers as a ready-to-go demonstration and a dynamic model for testing our approach.
The script automatically runs image generation and prediciton from start to finish for 5 samples per model, including:
Text processing, Image Generation + JSON files, Object Detection + Encoder, Similarity Checker and Predictions
