# Load the YOLOv5 model
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

for encoder_name, encoder in encoders.items():
  print(f"Using YOLOv5 with {encoder_name} Encoder\n{'-'*30}")

  #Set the encoder to evaluation mode
  encoder.eval()

  gold_labels = []
  predicted_labels = []
  sim_scores = []

  with open("SDv14.json") as f:
    my_dict = [json.loads(line) for line in f]
    mappe = []
    for i in my_dict[1680:]:
      img1 = i['img_gen1']
      img2 = i['img_gen2']
      label = i['label']
      path = i['img_local_path']
      gold_labels.append(label)
      mappe.append([img1, img2])

    print(mappe)
    for i in range(len(mappe)):
       feature_vectors = []
       for image_path in mappe[i]:
        img = cv2.imread(image_path)

        # Define image transforms
        # Run object detection on the image
        # (assuming you already have a YOLO model loaded and ready to use)
        results = yolov5(img)

        # Get the detected objects and their scores
        objects = results.pandas().xyxy[0]
        n_obj = len(objects)
        scores = objects['confidence'].tolist()

        # Select the top 10 objects based on their scores
        selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        selected_objects = objects.iloc[selected_indices]

        # Get the bounding boxes, labels, and confidence scores for the selected objects
        boxes = selected_objects[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = selected_objects['name'].tolist()
        scores = selected_objects['confidence'].tolist()

        for i in range(len(boxes)):
          box = boxes[i]
          label = labels[i]
          score = scores[i]
          
          # Draw the bounding box on the image
          cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
          
          # Add the label and confidence score to the bounding box
          text = f'{label}: {score:.2f}'
          cv2.putText(img, text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Define the fixed size for the cropped images
        fixed_size = (256,256)

        # Define image transforms
        transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        with torch.no_grad():
          entire_img = Image.fromarray(img) # Convert to PIL Image
          image_tensor1 = transform(entire_img).unsqueeze(0)
          features = encoder(image_tensor1).squeeze().numpy()
          #print("ENTIRE IMAGE FEATURES",features)
        feature_vectors.append(features)

        for i in range(len(selected_objects)):
          # Get the bounding box coordinates for the detected object
          xmin, ymin, xmax, ymax = selected_objects[['xmin', 'ymin', 'xmax', 'ymax']].values[i]

          # Crop the input image using the bounding box coordinates
          cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]

          # Resize the cropped image to a fixed size
          resized_img = cv2.resize(cropped_img, fixed_size)

          # Display the cropped image
          #cv2_imshow(resized_img)

          # Apply transforms to image
          resized_img = Image.fromarray(resized_img) # Convert to PIL Image
          image_tensor = transform(resized_img).unsqueeze(0)
        
          # Extract detected object features with densenet121
          with torch.no_grad():
              object_features = encoder(image_tensor).squeeze().numpy()

          feature_vectors.append(object_features)

    a = feature_vectors[0]
    b = feature_vectors[1]

    
    # Reshape the arrays to be 2D arrays with one row
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)

    # Compute the cosine similarity between the arrays
    similarity = cosine_similarity(a, b)

    # Compute the Structural Similarity Index (SSIM) between the arrays
    (score,diff) = ssim(np.squeeze(a), np.squeeze(b), full=True)

    sim_scores.append(similarity[0][0])

  print(scores)
  threshold = statistics.median(scores)
  print(threshold)

  print(f"Threshold for {encoder_name} is: ", threshold.round(2))
  for i in scores:
    if i < threshold.round(2):
      predicted_labels.append(1)
    else:
      predicted_labels.append(0)

  confusion_mat = confusion_matrix(gold_labels, predicted_labels)
  accuracy = accuracy_score(gold_labels, predicted_labels)
  precision = precision_score(gold_labels, predicted_labels)
  recall = recall_score(gold_labels, predicted_labels)
  f1 = f1_score(gold_labels, predicted_labels)

  print(f"Confusion Matrix:\n{confusion_mat}")
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1 Score: {f1}")
  print('\n')