# emoNet: Emotion Recognition from Images

Welcome to the GitHub repository for emoNet, a deep learning model designed for the task of emotion recognition from images. This model employs advanced image processing techniques and a neural network architecture to classify images into emotion categories such as happiness, sadness, anger, etc. Below you'll find complete instructions on preparing your dataset, training the emoNet model, and evaluating its performance.

## Prerequisites

Before you start, make sure you have the following prerequisites installed:

- Python 3.x
- PyTorch
- Torchvision
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

You can install these dependencies using pip:

```sh
pip install torch torchvision scikit-learn pandas matplotlib seaborn
```

## Dataset Preparation

Organize your image dataset into two main directories: one for training and another for testing. Each directory should contain subdirectories named according to the class labels of the emotions. For example:

```
dataset/
├── train/
│   ├── happiness/
│   ├── sadness/
│   └── anger/
└── test/
    ├── happiness/
    ├── sadness/
    └── anger/
```

## Configuration

Adjust training parameters by editing the `config.py` file in the emonet package. Key parameters include:

- `TRAIN_DIRECTORY`: Path to your training dataset.
- `TEST_DIRECTORY`: Path to your testing dataset.
- `TRAIN_SIZE` and `VAL_SIZE`: Fractions of the dataset used for training and validation.
- `BATCH_SIZE`: Number of samples per batch.
- `LR`: Learning rate.
- `NUM_OF_EPOCHS`: Number of epochs for training.

## Training the Model

Run the training script with the necessary arguments:

- `-m`, `--model`: Path to save the trained model file.
- `-p`, `--plot`: Graphic visualization of the model's loss and accuracy.

Example: 

```sh
python train_emonet.py --model model/emonet.pth --plot plots/training_plot.png
```

## Model Evaluation

To evaluate the model and generate classification scores for a set of images, follow these steps:

Organize Your Images: Place the images in a single folder. Supported formats include PNG, JPG, JPEG, BMP, and TIFF.

Run the Evaluation Script: Execute the script by providing the path to the images folder and the trained model file.

```sh
python evaluate_emonet.py /path/to/images/folder /path/to/trained/finalmodel.pth
```

Review the Output: The script will save the classification scores in a CSV file named classification_scores.csv within the images folder. The CSV file includes the filepath of each image and the classification scores for each emotion category.






