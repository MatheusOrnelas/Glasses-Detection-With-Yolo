
# YOLO Object Detection with Ultralytics

## 🎯 Introduction
This guide explains the implementation of the YOLO (You Only Look Once) object detection model using the Ultralytics library. The YOLO model is renowned for its speed and accuracy in detecting objects in images. This document covers the installation, setup, and usage of the YOLO model to perform object detection.

## ⚙ Installation and Setup
To set up the YOLO model for object detection, follow these steps:

1. Install the Ultralytics library:
   ```
   !pip install ultralytics
   ```

2. Clone the model configuration and weights:
   - The YAML file (`yolov8x.yaml`) contains the model configuration.
   - The weights file (`yolov8x.pt`) contains the pretrained model weights.

3. Prepare your dataset in a format compatible with YOLO, typically a YAML file (`data.yaml`). For this training was used Roboflow for image segmentation and generate data.yaml.

## ⚙ Training the Model
To train the YOLO model, use the following code:
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)
```
- `data='data.yaml'` specifies the dataset.
- `epochs=100` sets the number of training iterations.
- `imgsz=640` sets the image size for training.

## ⚙ Loading the Best Model
After training, load the best-performing model:
```python
model = YOLO('runs/detect/train/weights/best.pt')
```

## ⚙ Object Detection
Perform object detection on a list of images:
```python
# Images to predict
list_imgs = ['img_test_01.jpg', 'img_test_02.jpg', 'img_test_03.jpg', 'img_test_04.jpg']

# return a list of Results objects
results = model(list_imgs)  
```

## ⚙ Processing Results
Process the results to get the detection outputs:
```python
for result in results:
    boxes = result.boxes  # Bounding box outputs
    masks = result.masks  # Segmentation masks outputs
    keypoints = result.keypoints  # Pose outputs
    probs = result.probs  # Classification outputs
    result.show()  # Display to screen
    # result.save(filename='result_'+result.path.split('.')[0]+'.jpg')  # Save to disk
```

## 📝 Conclusion
This guide provides an overview of setting up and using the YOLO model for object detection. The YOLO model's efficiency in processing and accuracy makes it a powerful tool for image analysis tasks.


## 📌 Version
The version of project is 1.0.0.0.0

## ✒️ Autor
* **Developer** - *Developer of application* - [Matheus de Ornelas Vasconcellos](https://github.com/MatheusOrnelas)