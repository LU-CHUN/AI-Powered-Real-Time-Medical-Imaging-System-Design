import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import os


dataset_path = "D:/T&I/Dataset_BUSI_with_GT"


image_files = []
labels = []


for class_name in ['benign', 'malignant', 'normal']:
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        if '_mask' not in filename:
            image_path = os.path.join(class_path, filename)
            image_files.append(image_path)
            labels.append(class_name)


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
original_labels = label_encoder.classes_
labels = to_categorical(labels)


combined = list(zip(image_files, labels))
combined_train, combined_test = train_test_split(combined, test_size=0.3, random_state=42, stratify=[label for _, label in combined])


X_train, y_train = zip(*combined_train)
X_test, y_test = zip(*combined_test)


y_train = np.array(y_train)
y_test = np.array(y_test)


def load_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    return image


train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator()


def paths_to_data(image_paths, labels, datagen, batch_size=32):
    while True:
        batch_indices = np.random.choice(len(image_paths), size=batch_size)
        batch_data = [load_preprocess_image(image_paths[i]) for i in batch_indices]
        batch_labels = [labels[i] for i in batch_indices]
        batch_data = np.array(batch_data)
        batch_labels = np.array(batch_labels)
        yield datagen.flow(batch_data, batch_labels, batch_size=batch_size).next()


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


train_generator = lambda: paths_to_data(X_train, y_train, train_datagen)
test_generator = lambda: paths_to_data(X_test, y_test, test_datagen)

history = model.fit(
    x=train_generator(),
    steps_per_epoch=len(X_train) // 32,
    epochs=15,
    validation_data=test_generator(),
    validation_steps=len(X_test) // 32,
    callbacks=[early_stopping]
)


train_loss, train_accuracy = model.evaluate(train_generator(), steps=len(X_train) // 32)
test_loss, test_accuracy = model.evaluate(test_generator(), steps=len(X_test) // 32)

print('Training Accuracy:', train_accuracy, "Train loss", train_loss)
print('Testing Accuracy:', test_accuracy, "Test Loss", test_loss)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


model.save('breast_cancer_detection_model.h5')
