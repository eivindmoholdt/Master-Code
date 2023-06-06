# Detecting Out-of-Context Image-Caption Pairs in News: A Counter-Intuitive Method
Detecting Out-of-Context Image-Caption Pairs in News: A Counter-Intuitive Method

Under construction :)
### Requirements and pre-requisites:
All the code in this project was written and run in Google Colab. 
We recommend running all the code in this project in Google Colab.
The Python files has been written to allow easy automatic testing of our model. 
To test parts of the code we have included the raw IPYNB files in '/Google' folder.

Install the packages from requirements.txt using:
pip install -r 'requirements.txt'
Note: This also includes necesarry packages for the COSMOS model. However, package versions might not be the same. If you want to run the COSMOS model, see the COSMOS folder requirements.txt. and readme files.

DALL-E 2 (Open AI API) and Stable Diffusion (HuggingFace) setups are detailed in the respective python files in '/image_generation'

CLIP:
CLIP can be loaded by Git cloning git+https://github.com/openai/CLIP.git or via SentenceTransformers library
SentenceTransformers is included in requirements.txt.
SentenceTransformers library built on Pytorch allows for the easiest implementation:
model = SentenceTransformer('clip-ViT-L-14') or 
model = SentenceTransformer('clip-ViT-B-32')

See https://huggingface.co/sentence-transformers/clip-ViT-B-32
and https://huggingface.co/sentence-transformers/clip-ViT-L-14
for documentation

YOLOv5:
YOLOv5 is easily implemented with Pytorch
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained=True)
You can set pretrained=True or instead download the weights for the model and set path='path_to_weights'
Parameter 'yolov5' can be changed to 'yolov5s' or similar if you want specific model versions
See https://pytorch.org/hub/ultralytics_yolov5/ for more info

YOLOv7:
Clone YOLOv7 github rep with ! git clone https://github.com/WongKinYiu/yolov7.git
Import the custom function with:
from yolov7.hubconf import custom
yolov7 = custom('yolov7.pt') #add correct path to .pt file
after this YOLOv7 can be used the same way as YOLOv5

MASK-RCNN:
Loaded trough Detectron2
See Detectron2 docs for guide:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

## Image Generation
The Image Generation folder contains the Python files DALL-E-2_Gen.py and SD_Gen.py, for generating datasets with DALL-E 2 and SD.
Note that we originally load the captions from the original test_data.json file from COSMOS for image generation.
The modified captions we got post NER and censoring are saved in DALLE.json and/or SDv14.json files, you can load both original and mod captions from there as well
All details are outlined in the respective files

# Predictions and evaluation
