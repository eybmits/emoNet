import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from emonet import config as cfg, emoNet

# Define constants and configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/Users/markusbaumann/emotionrecognition/output/model82percent.pth'
FACE_CASCADE_PATH = '/Users/markusbaumann/emotionrecognition/preprocessing/haarcascade_frontalface_default.xml'
NORMALIZATION_MEAN = [0.5417377352714539]
NORMALIZATION_STD = [0.23748734593391418]

# Define transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3)), transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
])

# Initialize model
def load_model():
    model = emoNet(num_of_channels=1, num_of_classes=len(datasets.ImageFolder(cfg.TRAIN_DIRECTORY).classes))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

# CLAHE initialization
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Webcam and face detection
def start_webcam_detection(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_display(frame, face_cascade, model)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Detect and display emotions
def detect_and_display(frame, face_cascade, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        face_img_clahe = apply_clahe(face_img)

        face_tensor = preprocess(face_img_clahe)
        predicted_class = predict_emotion(face_tensor, model)
        draw_predictions(frame, predicted_class, x, y, w, h)

    cv2.imshow('Face Emotion Recognition', frame)

# Preprocess face image
def preprocess(face_img):
    face_img = np.expand_dims(face_img, axis=(0, 1)) / 255.0
    return torch.from_numpy(face_img).type(torch.FloatTensor).to(DEVICE)

# Predict emotion
def predict_emotion(face_tensor, model):
    with torch.no_grad():
        output = model(face_tensor)
        predicted = torch.max(F.softmax(output, dim=1), 1)[1]
    return datasets.ImageFolder(cfg.TRAIN_DIRECTORY).classes[predicted.item()]

# Draw predictions on frame
def draw_predictions(frame, predicted_class, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, predicted_class, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

if __name__ == "__main__":
    model = load_model()
    start_webcam_detection(model)
