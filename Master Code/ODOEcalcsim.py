#CalcSim Functions using Object Detection Model + Object Encoder


import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import config
from config import *
from sklearn.metrics.pairwise import cosine_similarity
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

#Get Model choice from Config
def set_detection_model():
  if config.YOLOV5:
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model
  if config.YOLOV7:
    from yolov7.hubconf import custom
    model = custom('yolov7.pt')
    return model
  if config.MASKRCNN:
    # Set up Detectron2 configuration
    setup_logger()

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    model = DefaultPredictor(cfg)
    return model

#YOLO
def calc_sim_yolo(image_pairs):
  feature_vectors = []

  for image_path in image_pairs:
    img = cv2.imread(image_path)
    #cv2_imshow(img)

    # Run object detection on the image
    model = set_detection_model()
    results = model(img)

    # Get the detected objects
    objects = results.pandas().xyxy[0]
    n_obj = len(objects)

    # Get the bounding boxes, labels, and confidence scores for the detected objects
    boxes = objects[['xmin', 'ymin', 'xmax', 'ymax']].values
    labels = objects['name'].tolist()
    scores = objects['confidence'].tolist()

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = scores[i]
        
        # Draw the bounding box on the image
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        # Add the label and confidence score to the bounding box
        text = f'{label}: {score:.2f}'
        cv2.putText(img, text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    #cv2_imshow(img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Define the fixed size for the cropped images
    fixed_size = (256,256)

    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Iterate over each detected object
    if n_obj == 0:
      #print("no objects detected")
      # Extract features with ResNet50
      with torch.no_grad():
        img = Image.fromarray(img) # Convert to PIL Image
        image_tensor = transform(img).unsqueeze(0)
        features = encoders(image_tensor).squeeze().numpy()
        #print("ENTIRE IMAGE FEATURES",features)
      feature_vectors.append(features)
    else:
      for i in range(len(objects)):
        # Get the bounding box coordinates for the detected object
        xmin, ymin, xmax, ymax = objects[['xmin', 'ymin', 'xmax', 'ymax']].values[i]

        # Crop the input image using the bounding box coordinates
        cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Resize the cropped image to a fixed size
        resized_img = cv2.resize(cropped_img, fixed_size)

        # Display the cropped image
        #cv2_imshow(resized_img)

        # Apply transforms to image
        resized_img = Image.fromarray(resized_img) # Convert to PIL Image
        image_tensor = transform(resized_img).unsqueeze(0)
      

        # Extract features with ResNet50
        with torch.no_grad():
            features = encoders(image_tensor).squeeze().numpy()

        feature_vectors.append(features)

  a = feature_vectors[0]
  b = feature_vectors[1]

  import numpy as np
  from sklearn.metrics.pairwise import cosine_similarity
  # reshape the arrays to be 2D arrays with one row
  a = a.reshape(1, -1)
  b = b.reshape(1, -1)

  # compute the cosine similarity between the arrays
  similarity = cosine_similarity(a, b)

  return similarity[0][0]

#MASK-RCNN 
def calc_sim_maskrcnn(image_path):

  model = set_detection_model()

  image = Image.open(image_path)
  image = np.array(image)
  outputs = model(image)
  
  feature_vectors = outputs["instances"].pred_boxes.tensor.cpu().numpy().squeeze()

  a = feature_vectors[0]
  b = feature_vectors[1]

  # Reshape the feature vectors to be 1D arrays
  a = np.reshape(a, (1, -1))
  b = np.reshape(b, (1, -1))

  # compute the cosine similarity between the arrays
  similarity = cosine_similarity(a, b)
  return similarity[0][0]