import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from emonet import emoNet  # Make sure this matches your actual import
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the emotion detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happiness", 4: "Sadness", 5: "Surprise"}
model = emoNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model.load_state_dict(torch.load('/Users/markusbaumann/emotionrecognition/output/model.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing transformations
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Specify the folder containing images
image_folder = 'img'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
num_images = len(image_files)

# Adjust figure size based on the number of images
plt.figure(figsize=(15, num_images * 2))  # Adjust width to 15 for better visibility

# Initialize subplot index
subplot_index = 1

# Process and plot each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_file} not found, skipping.")
        continue

    # Apply preprocessing directly to the whole image
    transformed_image = data_transform(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    transformed_image = transformed_image.unsqueeze(0).to(device)
    transformed_image.requires_grad_(True)

    # Predict emotion and generate gradients
    with torch.enable_grad():
        predictions = model(transformed_image)
        prob = F.softmax(predictions, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        emotion = emotion_dict[top_class.item()]
        predictions[:, top_class.item()].backward()
        gradients = transformed_image.grad.data.abs().squeeze().cpu().numpy()
        saliency_map = np.clip(gradients * 255, 0, 255).astype(np.uint8)
        saliency_map = cv2.resize(saliency_map, (image.shape[1], image.shape[0]))

    # Apply heatmap to the original image
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Plot Original Image
    plt.subplot(num_images, 3, subplot_index)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    # Plot Saliency Map
    plt.subplot(num_images, 3, subplot_index + 1)
    plt.imshow(saliency_map, cmap='gray')
    plt.title('Saliency Map')
    plt.axis('off')

    # Plot Superimposed Image
    plt.subplot(num_images, 3, subplot_index + 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Emotion: {emotion}\nConfidence: {top_p.item() * 100:.2f}%')
    plt.axis('off')

    # Increment subplot index by 3 for the next set of images
    subplot_index += 3

plt.tight_layout()
plt.show()
