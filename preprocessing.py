import cv2
import os
from pathlib import Path

def apply_clahe_to_directory(input_directory, output_directory):
    """
    Wenden Sie CLAHE auf alle Bilder in einem Verzeichnis an und speichern Sie sie im entsprechenden Ausgabeverzeichnis.
    """
    # Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Durchlaufen Sie alle Bilder im Verzeichnis
    for img_name in os.listdir(input_directory):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Bild laden und in Graustufen konvertieren
            image_path = os.path.join(input_directory, img_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # CLAHE anwenden
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_clahe = clahe.apply(image)
            
            # Bearbeitetes Bild im Ausgabeverzeichnis speichern
            cv2.imwrite(os.path.join(output_directory, img_name), image_clahe)

def apply_clahe_to_emotion_folders(parent_input_path, parent_output_path):
    """
    Wenden Sie CLAHE auf alle Bilder in den Emotionsunterverzeichnissen an.
    """
    # Durchlaufen Sie jedes Unterordner im Hauptverzeichnis
    for emotion_folder in os.listdir(parent_input_path):
        emotion_input_path = os.path.join(parent_input_path, emotion_folder)
        emotion_output_path = os.path.join(parent_output_path, emotion_folder)
        
        if os.path.isdir(emotion_input_path):
            print(f"Verarbeite {emotion_folder} Bilder...")
            apply_clahe_to_directory(emotion_input_path, emotion_output_path)


# Pfade zu Ihren Eingabeverzeichnissen
train_input_path = '/Users/markusbaumann/emotionrecognition/data/validation_set'
test_input_path = '/Users/markusbaumann/emotionrecognition/data/validation_set'

# Pfade zu den Ausgabeverzeichnissen, in denen die bearbeiteten Bilder gespeichert werden
train_output_path = '/Users/markusbaumann/emotionrecognition/data/validation_set2'
test_output_path = '/Users/markusbaumann/emotionrecognition/data/validation_set2'

apply_clahe_to_emotion_folders(train_input_path, train_output_path)
apply_clahe_to_emotion_folders(test_input_path, test_output_path)

print("CLAHE Anwendung auf alle Bilder abgeschlossen.")