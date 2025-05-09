import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pickle
import os
import random

# Directory paths
path = "TrafficSign/traffic_Data/DATA"  # Directory containing class subdirectories (0, 1, 2, etc.)
labelFile = "TrafficSign/labels.csv"  # CSV file containing class labels
batch_size_val = 50
steps_per_epoch_val = 50
epochs_val = 30
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

# Prepare data arrays
images = []
classNo = []

# Load the labels CSV file
labels_df = pd.read_csv(labelFile)

# Get all subdirectories (classes) in the dataset
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes...")

# Loop through each class folder (0, 1, 2, ..., etc.)
for count in range(noOfClasses):
    classFolderPath = os.path.join(path, str(count))  # Path for each class folder (e.g., "0", "1", ...)

    if os.path.exists(classFolderPath):
        myPicList = os.listdir(classFolderPath)  # List images in the class folder
        for y in myPicList:
            curImg = cv2.imread(os.path.join(classFolderPath, y))  # Read each image
            if curImg is not None:  # Ensure the image is read correctly
                curImg = cv2.resize(curImg, (32, 32))  # Resize all images to a fixed size (32x32)
                images.append(curImg)
                classNo.append(count)  # Assign the class number
        print(f"Class {count} imported.")
    else:
        print(f"Class {count} folder not found.")  # Handle missing class folders

# Convert images and class numbers to numpy arrays
images = np.array(images)
classNo = np.array(classNo)

# Split dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Check the shapes of the datasets
print("Data Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# Ensure dimensions are correct
assert (X_train.shape[0] == y_train.shape[0])
assert (X_validation.shape[0] == y_validation.shape[0])
assert (X_test.shape[0] == y_test.shape[0])
assert (X_train.shape[1:] == imageDimensions)
assert (X_validation.shape[1:] == imageDimensions)
assert (X_test.shape[1:] == imageDimensions)


# Preprocessing functions (convert to grayscale and equalize)
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize the image to range [0, 1]
    return img


# Apply preprocessing to all data (train, validation, and test sets)
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape the data to match the model's expected input format
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation to improve model generalization
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=10)
dataGen.fit(X_train)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


# Define the Convolutional Neural Network (CNN) model
def myModel():
    no_Of_Filters = 60
    size_Of_Filter = (5, 5)
    size_Of_Filter2 = (3, 3)
    size_Of_Pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_Of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                     activation='relu'))
    model.add(Conv2D(no_Of_Filters, size_Of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_Of_Pool))
    model.add(Conv2D(no_Of_Filters // 2, size_Of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, size_Of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_Of_Pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the model
model = myModel()
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                    validation_data=(X_validation, y_validation))

# Plot the training and validation loss and accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])

# Save the trained model to a file
with open("model_trained.p", "wb") as pickle_out:
    pickle.dump(model, pickle_out)
