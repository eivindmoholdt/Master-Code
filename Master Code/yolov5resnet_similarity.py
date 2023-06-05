def calc_sim(image_pairs):
  feature_vectors = []

  for image_path in image_pairs:
    img = cv2.imread(image_path)
    #cv2_imshow(img)

    # Run object detection on the image
    # (assuming you already have a YOLO model loaded and ready to use)
    results = yolov5(img)

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

    # Initialize ResNET model and set to eval mode
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np


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
        features = resnet50(image_tensor).squeeze().numpy()
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
              features = resnet50(image_tensor).squeeze().numpy()

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
