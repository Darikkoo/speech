import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = r"C:\Users\WEB\Desktop\mnist_png"
train_dir = os.path.join(DATASET_PATH, "training")
test_dir = os.path.join(DATASET_PATH, "testing")

datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=128,
    class_mode="sparse"
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=128,
    class_mode="sparse"
)

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = create_model()
model.fit(train_generator, epochs=5, validation_data=test_generator)

model.save("mnist_png_model.h5")

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w * h > 100:
            digit = thresh[y:y+h, x:x+w]
            resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0
            digits.append(normalized)
            boxes.append((x, y, w, h))

    return digits, boxes, image

def recognize_digits(model, digits):
    predictions = []

    for digit in digits:
        digit = digit.reshape(1, 28, 28, 1)
        prediction = model.predict(digit)
        predicted_class = np.argmax(prediction)
        predictions.append(predicted_class)

    return predictions

image_path = "find.jpg"
output_path = "outpt.png"

digits, boxes, image = preprocess_image(image_path)
predictions = recognize_digits(model, digits)
