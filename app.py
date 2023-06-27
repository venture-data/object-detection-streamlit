# Import required libraries
import cv2
import numpy as np
import streamlit as st
import requests
import tempfile
import os

# Streamlit app
st.title("Object Detection using YOLOv3")

# Load YOLOv3 model and configuration files
# net = cv2.dnn.readNet("D:/Statifi/object detection yolo/yolov3.weights", "D:/Statifi/object detection yolo/yolov3.cfg")


#############################
# URL for the model architecture file on GitHub
model_architecture_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'

# URL for the model weights file on GitHub
model_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'

@st.cache(allow_output_mutation=True)
def load_model(model_architecture_path, model_weights_path):
    net = cv2.dnn.readNetFromDarknet(model_architecture_path, model_weights_path)
    return net

# Define the file paths for model architecture and weights
model_architecture_path = "yolov3.cfg"
model_weights_path = "yolov3.weights"

# Check if the files exist, if not download them
if not os.path.exists(model_architecture_path):
    st.write("Downloading model architecture file...")
    response = requests.get(model_architecture_url)
    with open(model_architecture_path, "w") as f:
        f.write(response.text)

if not os.path.exists(model_weights_path):
    st.write("Downloading model weights file...")
    response = requests.get(model_weights_url)
    with open(model_weights_path, "wb") as f:
        f.write(response.content)

# Load the model
net = load_model(model_architecture_path, model_weights_path)
#############################




# Define classes to be detected
classes = []
with open("D:/Statifi/object detection yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set minimum confidence threshold and non-maximum suppression threshold
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.4, 0.01)

# Load image and display
img_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if img_path is not None:
    # Load image
    img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

    # Set input for the neural network
    net.setInput(blob)

    # Get output layer names
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Forward pass through the neural network
    outputs = net.forward(output_layers_names)

    # Create bounding boxes and confidence scores for detected objects
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                w = int(detection[2] * img.shape[1])
                h = int(detection[3] * img.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Draw bounding boxes and labels for the detected objects
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    # Show the image with detected objects
    st.image(img, channels="BGR")
