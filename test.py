import numpy as np
import cv2
import pickle  # Import pickle for loading the model
import pandas as pd

# Setup for webcam
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.6
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model (make sure it's the .p file)
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Load the labels CSV file
labelFile = "TrafficSign/labels.csv"
labels_df = pd.read_csv(labelFile)

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # Normalize the image
    return img

# Class name mapping function
def getClassName(classNo):
    try:
        return labels_df.iloc[classNo]['Name']  # Get the 'Name' of the class from the CSV file
    except:
        return "Unknown Class"

# Main loop to process webcam frames
while True:
    success, imgOriginal = cap.read()

    # Resize and preprocess the image
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)

    # Display the processed image
    cv2.imshow("Processed Image", img)

    # Prepare image for prediction
    img = img.reshape(1, 32, 32, 1)  # Add batch dimension

    # Add text for predictions on the original frame
    cv2.putText(imgOriginal, "Class: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "Probability", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Get model predictions
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Get the class index
    probabilityValue = np.max(predictions)  # Get the probability value

    # If prediction confidence is greater than the threshold, display it
    if probabilityValue > threshold:
        className = getClassName(classIndex)  # Fetch the class name from the CSV
        cv2.putText(imgOriginal, f"{classIndex} {className}", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the result in a window
    cv2.imshow("Result", imgOriginal)

    # Exit condition: press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
