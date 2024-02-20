# /path/to/full_script.py
import os
import torch
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms.functional import to_tensor, resize
from emonet import config as cfg, EarlyStopping, LRScheduler, emoNet
import os
import torch
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms.functional import to_tensor, resize
from emonet import config as cfg, EarlyStopping, LRScheduler, emoNet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from transferoriginial import TransferLearningModel

# Function to normalize image saturation and brightness
def normalize_saturation_and_brightness(image):
    image_hsv = image.convert('HSV')
    h, s, v = image_hsv.split()
    s = ImageOps.autocontrast(s)
    v = ImageOps.autocontrast(v)
    image_normalized = Image.merge('HSV', (h, s, v)).convert('RGB')
    return image_normalized


# Custom Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, label_mapping=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [file for file in sorted(os.listdir(img_dir)) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name)
        image = normalize_saturation_and_brightness(image)
        label_text = self.extract_label(self.images[idx])
        label = self.label_mapping.get(label_text, -1)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def extract_label(file_name):
        return file_name.split('_')[-1].split('.')[0]

# Label mapping and transform placeholders
label_mapping = {
    'anger': 0, 'disgust': 1, 'fear': 2,
    'happiness': 3, 'sadness': 4, 'surprise': 5
}

# Set directory for validation set
directory = '/Users/markusbaumann/emotionrecognition/data/validation_set'

# Define transformations matching the training script
from torchvision import transforms

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with the specified mean and std
])


# Instantiate dataset and dataloader for validation set
val_dataset = CustomImageDataset(img_dir=directory, transform=val_transforms, label_mapping=label_mapping)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load the model and prepare for evaluation
model = TransferLearningModel(num_classes=6)
model.load_state_dict(torch.load('/Users/markusbaumann/emotionrecognition/output/final_model.pth', map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model on the validation set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')


all_labels = []
all_predictions = []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate and print accuracy
accuracy = 100 * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
print(f'Validation Accuracy: {accuracy:.2f}%')

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_names = list(label_mapping.keys())

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()