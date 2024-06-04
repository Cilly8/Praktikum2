import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle
import random
import os
from sklearn.utils import shuffle

"""
# Get the training and testing dataset
with open(os.path.join("dataset", "train.p"), mode='rb') as training_data:
    train = pickle.load(training_data)
with open(os.path.join("dataset", "valid.p"), mode='rb') as validation_data:
    valid = pickle.load(validation_data)

# Get the features and labels of the datasets
# The features are the images of the signs
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']


print("Number of training examples: ", X_train.shape[0])
print("Number of validation examples: ", X_valid.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", len(np.unique(y_train)))


# Plot a random picture from the training dataset
i = np.random.randint(1, len(X_train))
plt.grid(False)
plt.imshow(X_train[i])
print("Label: ", y_train[i])

# Plot (width x height) pictures from the training dataset
grid_width = 5
grid_height = 4

fig, axes = plt.subplots(grid_height, grid_width, figsize = (10,10))
axes = axes.ravel()

for i in np.arange(0, grid_width * grid_height):
    index = np.random.randint(0, len(X_train))
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index], fontsize = 15)
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.3)

# Plotting histograms of the count of each sign
def histogram_plot(dataset: np.ndarray, label: str):
    # Plots a histogram of the dataset

    #Args:
    #    dataset: The input data to be plotted as a histogram.
    #    label: The label of the histogram.
    hist, bins = np.histogram(dataset, bins=43)
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()

histogram_plot(y_train, "Training examples")
histogram_plot(y_valid, "Validation examples")

# Daten mischen
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

# Bilder normalisieren
X_train_norm = X_train / 255.0
X_valid_norm = X_valid / 255.0

# Überprüfen, ob die Bilder korrekt konvertiert und normalisiert wurden
i = random.randint(1, len(X_train_norm))
plt.grid(False)
plt.imshow(X_train[i])
plt.figure()
plt.grid(False)
plt.imshow(X_train_norm[i].squeeze(), cmap='gray')


A list of all classes:
- 0 = Speed limit (20km/h)
- 1 = Speed limit (30km/h)
- 2 = Speed limit (50km/h)
- 3 = Speed limit (60km/h)
- 4 = Speed limit (70km/h)
- 5 = Speed limit (80km/h)
- 6 = End of speed limit (80km/h)
- 7 = Speed limit (100km/h)
- 8 = Speed limit (120km/h)
- 9 = No passing
- 10 = No passing for vehicles over 3.5 metric tons
- 11 = Right-of-way at the next intersection
- 12 = Priority road
- 13 = Yield
- 14 = Stop
- 15 = No vehicles
- 16 = Vehicles over 3.5 metric tons prohibited
- 17 = No entry
- 18 = General caution
- 19 = Dangerous curve to the left
- 20 = Dangerous curve to the right
- 21 = Double curve
- 22 = Bumpy road
- 23 = Slippery road
- 24 = Road narrows on the right
- 25 = Road work
- 26 = Traffic signals
- 27 = Pedestrians
- 28 = Children crossing
- 29 = Bicycles crossing
- 30 = Beware of ice/snow
- 31 = Wild animals crossing
- 32 = End of all speed and passing limits
- 33 = Turn right ahead
- 34 = Turn left ahead
- 35 = Ahead only
- 36 = Go straight or right
- 37 = Go straight or left
- 38 = Keep right
- 39 = Keep left
- 40 = Roundabout mandatory
- 41 = End of no passing
- 42 = End of no passing by vehicles over 3.5 metric tons
"""

def preprocess_image(image_path):
    # Lade das Bild
    image = tf.io.read_file(image_path)
    # Dekodiere das Bild (PNG oder JPG)
    image = tf.image.decode_image(image, channels=3)
    # Ändere die Größe des Bildes
    image = tf.image.resize(image, [300, 300])
    # Normalisiere die Pixelwerte auf den Bereich [0, 1]
    image = image / 255.0
    return image

def preprocess_images_from_folder(folder_path):
    images = []
    # Iteriere über alle Dateien im Ordner
    for filename in os.listdir(folder_path):
        # Erstelle den vollständigen Pfad zur Bilddatei
        image_path = os.path.join(folder_path, filename)
        # Stelle sicher, dass es sich um eine Bilddatei handelt (optional)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Vorverarbeite das Bild und füge es der Liste hinzu
            image = preprocess_image(image_path)
            images.append(image)
    return images

# Pfad für den Ordner mit den Bildern:
folder_path = 'venv/signLibrary'
x_own, y_own = preprocess_images_from_folder(folder_path)

# Anzeigen der Anzahl vorverarbeiteter Bilder
print(f"Anzahl der vorverarbeiteten Bilder: {len(x_own)}")

# Anzahl der Trainings- und Validierungsbeispiele
print("Number of training examples: ", x_own.shape[0])
print("Image data shape =", x_own[0].shape)
print("Number of classes =", len(np.unique(y_own)))

# Zufälliges Bild aus dem Trainingsdatensatz anzeigen
i = np.random.randint(1, len(x_own))
plt.grid(False)
plt.imshow(x_own[i])
print("Label: ", y_own[i])

# (Breite x Höhe) Bilder aus dem Trainingsdatensatz anzeigen
grid_width = 5
grid_height = 4

fig, axes = plt.subplots(grid_height, grid_width, figsize = (10,10))
axes = axes.ravel()

for i in np.arange(0, grid_width * grid_height):
    index = np.random.randint(0, len(x_own))
    axes[i].imshow(x_own[index])
    axes[i].set_title(y_own[index], fontsize = 15)
    axes[i].axis('off')

plt.subplots_adjust(hspace = 0.3)

# Histogramme der Anzahl der jeweiligen Zeichen anzeigen
def histogram_plot(dataset: np.ndarray, label: str):
    # Plot eines Histogramms des Datensatzes

    # Argumente:
    #    dataset: Die Eingabedaten, die als Histogramm geplottet werden.
    #    label: Das Label des Histogramms.
    hist, bins = np.histogram(dataset, bins=43)
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel(label)
    plt.ylabel("Image count")
    plt.show()

histogram_plot(y_own, "Own examples")

# Daten mischen
x_own, y_own = shuffle(x_own, y_own)

# Bilder sind bereits normalisiert durch die Vorverarbeitung der Preprocessing-Funktion

# Überprüfen, ob die Bilder korrekt konvertiert und normalisiert wurden
i = random.randint(1, len(x_own))
plt.grid(False)
plt.imshow(x_own[i])
plt.figure()
plt.grid(False)
plt.imshow(x_own[i].numpy(), cmap='gray')