from pathlib import Path
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path.cwd() #sets DATA_DIR to current folder
IMAGE_DIR = Path.cwd() / "Images" #only used for DALL-E
IMAGE_DIR.mkdir(exist_ok=True) #only used for DALL-E
TARGET_DIR = Path.cwd() / "Responses" #only used for DALL-E

#Set API key.
#Sign up for the OpenAI API and create a new API key by clicking on the dropdown menu on your profile and selecting View API keys
#Add your OpenAI API key found at https://platform.openai.com/account/api-keys
OPENAI_API_KEY=""


#Choose text-to-image model. Do not run both models at the same time
# Set both to False to not generate any images
DALLE2 = False
SD = False

#Use Object Detection model?
USE_DETECTION = False

# If set true, choose Object Detection Model
YOLOV5 = False
YOLOV7 = False
MASKRCNN = False


# Object Encoder Model
#You can test all different object encoders or set a specific one by commenting out models in encoders dict
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from sentence_transformers import SentenceTransformer, util



# Load pre-trained object encoders
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
densenet169 = models.densenet169(pretrained=True)
efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
CLIPl14 = SentenceTransformer('clip-ViT-L-14')
CLIPb32 = SentenceTransformer('clip-ViT-B-32')

encoders = {
  'CLIPl14':CLIPl14,
  'CLIPb32':CLIPb32,
  'ResNet18': resnet18,
  'ResNet50': resnet50,
  'ResNext': resnext50_32x4d,
  'DenseNet121': densenet121,
  'DenseNet169': densenet169,
  'EfficientNet': efficientnet,
}

#The similarity threshold for OOC/NOOC predictions. Sim scores below the threshold are predicted OOC, above are NOOC
#Note that the threshold values differ from each object encoder and each combination of object detection model + object encoder
threshold = 0.50



