from pathlib import Path
import torch

#DATA_DIR = Path.cwd() / "MyNewDataDir"
#DATA_DIR.mkdir(exist_ok=True)
DATA_DIR = Path.cwd() #sets DATA_DIR to current folder
IMAGE_DIR = Path.cwd() / "Images"
IMAGE_DIR.mkdir(exist_ok=True)
TARGET_DIR = Path.cwd() / "Responses" #only used for DALL-E

#Add your OpenAI API key found at https://platform.openai.com/account/api-keys
OPENAI_API_KEY=""

#Choose text-to-image model. Do not run both models at the same time
DALLE = True
SD = False

#Text Processing
text_processing_NER = True
censoring = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



