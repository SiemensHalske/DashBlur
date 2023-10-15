# ****** IMPORTS ******

# Standard Libraries
import os
import sys
import numpy as np

# EMNIST Dataset
import emnist

# TensorFlow and Keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras import (
    Model, Input
)
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, concatenate, AveragePooling2D,
    Activation, add
)
from keras.optimizers import (
    Adam, RMSprop
)
from keras.regularizers import (
    l1, l2, l1_l2
)
from keras.callbacks import (
    Callback, LambdaCallback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
)

# Scikit-learn
from sklearn.metrics import classification_report


# ****** LIMITS & SEEDS ******

TRAINING_SAMPLES = 100000  # int(sys.argv[1])
TEST_SAMPLES = 5000  # int(sys.argv[2])
BATCH_SIZE = 32  # int(sys.argv[3])
EPOCHS = 20  # int(sys.argv[4])

L1_REG = 0.001
L2_REG = 0.001
L1_L2_REG = 0.005

np.random.seed(42)
tf.random.set_seed(42)

# Sample Python code for training a model using EMNIST data


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nEpoch {epoch+1} has finished! - Loss: {logs['loss']}, Accuracy: {logs['accuracy']}\n**********")


# Define learning rate schedule function
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Load EMNIST data
train_images, train_labels = emnist.extract_training_samples('byclass')
test_images, test_labels = emnist.extract_test_samples('byclass')

# Use a smaller subset of data to prevent memory issues
train_images = train_images[:TRAINING_SAMPLES]
train_labels = train_labels[:TRAINING_SAMPLES]
test_images = test_images[:TEST_SAMPLES]
test_labels = test_labels[:TEST_SAMPLES]

print(f"Train label shape: {train_labels.shape}")
print(f"Test label shape: {test_labels.shape}")

# Preprocess the data
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

train_labels = to_categorical(train_labels, num_classes=62)
# train_labels = np.argmax(train_labels, axis=1)
test_labels = to_categorical(test_labels, num_classes=62)

input_img = Input(shape=(28, 28, 1))

# First conv block
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(
    L1_REG), bias_regularizer=l1_l2(L1_L2_REG))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Second conv block with residual connection
y = Conv2D(64, (3, 3), padding='same')(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(64, (3, 3), padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

# Match the channels for residual connection
# Use 1x1 conv to increase channels from 32 to 64
x_match = Conv2D(64, (1, 1), padding='same')(x)
x_match = BatchNormalization()(x_match)

# Actual Residual Connection
y = add([y, x_match])

# Inception Module
tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(y)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(y)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(y)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(64, (1, 1), padding='same', activation='relu')(y)

y_2 = concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)
y_2 = BatchNormalization()(y_2)
y_2 = AveragePooling2D((2, 2))(y_2)

# Fully connected layers
z = Flatten()(y_2)
z = Dense(128, activation='relu', kernel_regularizer=l2(
    L2_REG), bias_regularizer=l2(L2_REG))(z)
z = Dropout(0.5)(z)
z = Dense(64, activation='relu', kernel_regularizer=l2(
    L2_REG), bias_regularizer=l2(L2_REG))(z)
z = Dropout(0.4)(z)  # Slightly reduced dropout
output = Dense(62, activation='softmax')(z)  # 62 classes for EMNIST

# Create the model
model = Model(inputs=input_img, outputs=output)

optimizer_adam = Adam(
    learning_rate=0.0005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08
)

optimizer_rmsprop = RMSprop(
    learning_rate=0.0001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Initialize the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation accuracy
    patience=3,          # Number of epochs with no improvement to wait before stopping
    restore_best_weights=False  # Restore the best weights when stopped
)

# Initialize the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Initialize the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(schedule, verbose=1)


# Compile the model
model.compile(optimizer=optimizer_rmsprop,
              loss='categorical_crossentropy', metrics=['accuracy'])

# Create a callback to print the loss and accuracy after each epoch
print_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
    f"Epoch {epoch + 1} has finished! - Loss: {logs['loss']}, Accuracy: {logs['accuracy']}"))

# print(f"Shape of train_images: {train_images.shape}")
# print(f"Shape of train_labels: {train_labels.shape}")

try:
    # Train the model
    model.fit(train_images, train_labels, epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[print_callback, early_stopping, lr_scheduler],
              validation_data=(test_images, test_labels))
except KeyboardInterrupt:
    print("Training has been stopped early!")
    exit()
except Exception as e:
    print(e)
    # shutdown ubuntu
    # os.system("shutdown now -h")
    exit()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")
summary = model.summary()
print(summary)

# Predict classes using the test set
y_pred = model.predict(test_images, batch_size=16, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

# Convert one-hot encoded test_labels to single labels
y_test_single_label = np.argmax(test_labels, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_single_label, y_pred_bool))

# Save the model (optional)
save_path = "/home/hendrik/Documents/Projects/DashBlur/src/"
model.save(save_path + "emnist_model.h5")
