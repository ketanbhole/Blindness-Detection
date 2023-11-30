from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load train labels for year '19'
train_19_labels = pd.read_csv(r'E:\blind\archive\labels\trainLabels19.csv')
test_19_labels = pd.read_csv(r'E:\blind\archive\labels\testImages19.csv')


# Load train images for year '19'
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img = img.resize((128, 128))  # Resize images to fit the model
        img_array = np.array(img)
        if img_array is not None:
            images.append(img_array)
    return images


train_19_images = load_images_from_folder(r'E:\blind\resized train 19')
test_19_images = load_images_from_folder(r'E:\blind\resized test 19')

# Merge data and labels for year '19'
train_images = np.array(train_19_images)
test_images = np.array(test_19_images)

# Assuming 'diagnosis' is the column name for labels in the train CSV file for year '19'
train_labels = np.array(train_19_labels['diagnosis'].tolist())

# Preprocessing
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Encode labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
num_classes = len(encoder.classes_)
train_labels = to_categorical(train_labels, num_classes=num_classes)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# Add other layers as needed...

# Add Flatten and Dense layers to match the number of classes
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))  # Output layer with appropriate number of units

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_val, y_val))

# Save the trained model
model.save('blindness_detection_model.h5')  # Save the model for later use


# Example usage of the trained model for prediction
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize to match the input size of the model
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_blindness(image_path):
    trained_model = load_model('blindness_detection_model.h5')  # Load the saved model

    # Preprocess the image
    processed_img = preprocess_image(image_path)

    # Make prediction
    prediction = trained_model.predict(processed_img)

    # Decode the prediction to get the predicted severity level
    severity_levels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    predicted_severity_index = np.argmax(prediction)
    predicted_severity = severity_levels[predicted_severity_index]

    return predicted_severity


# Example usage:
image_path = r'E:\blind\archive\resized train 19\0e0fc1d9810c.jpg'  # Replace with the user's image path
predicted_severity = predict_blindness(image_path)
print(f"The predicted severity of diabetic retinopathy for the image is: {predicted_severity}")
