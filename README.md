# emoNet: Emotion Recognition from Images

Welcome to the GitHub repository for emoNet, a cutting-edge deep learning model designed for the task of emotion recognition from images. This model employs advanced image processing techniques and a neural network architecture to classify images into emotion categories such as happiness, sadness, anger, etc. Below you'll find complete instructions on preparing your dataset, training the emoNet model, and evaluating its performance.

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

To evaluate the effectiveness of the model and assign classification metrics to a new set of images, proceed with the steps detailed below:

1. **Organize Your Data:** Position the evaluation images in a distinct catalog. Acceptable formats comprise PNG, JPG, JPEG, BMP, and TIFF.

2. **Facilitate the Analysis Script:** Operationalize the file by delineating the data's gift of locale in addition to the box storing the elevation of enlightenment.

```sh
python evaluate_emonet.py /path/to/images/folder /path/to/trained/finalmodel.pth
```

3. **Reflect on the Record:** Residually, a paper encoded in the semblance of a CSV file will be chronicled within the source's gallery, this register bequeathed with the readings under the insight of 'classification_scores.csv' will incorporate the name and chronicle of each portraiture vis-à-vis the proportioned authorizations for the suggested emotional limitations.




