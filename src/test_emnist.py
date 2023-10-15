# ****** IMPORTS ******
import os
import numpy as np
import sys
import emnist
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import to_categorical
# from tensorflow.keras import layers, models
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1, l2, l1_l2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate, AveragePooling2D
from keras.callbacks import LambdaCallback
from keras.layers import Activation, add
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
# from keras.datasets import mnist
# import numpy as np


# ****** LIMITS & SEEDS ******

TRAINING_SAMPLES = 100_000  # int(sys.argv[1])
TEST_SAMPLES = 10_000  # int(sys.argv[2])
BATCH_SIZE = 16  # int(sys.argv[3])
EPOCHS = 100  # int(sys.argv[4])

L1_REG = 0.0025
L2_REG = 0.0025
L1_L2_REG = 0.001

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
        return lr * 0.9


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
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(
    L1_REG), bias_regularizer=l2(L2_REG))(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(
    L1_REG), bias_regularizer=l1(L2_REG))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l1(
    L1_REG), bias_regularizer=l1(L2_REG))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Second conv block with residual connection
y = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(
    L1_REG), bias_regularizer=l2(L2_REG))(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l1(
    L1_REG), bias_regularizer=l2(L2_REG))(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

# Match the channels for residual connection
# Use 1x1 conv to increase channels from 32 to 64
x_match = Conv2D(64, (1, 1), padding='same')(x)
x_match = BatchNormalization()(x_match)

# Actual Residual Connection
# y = add([y, x_match])

# Inception Module
tower_1 = Conv2D(256, (1, 1), padding='same', activation='relu')(y)
tower_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(256, (1, 1), padding='same', activation='relu')(y)
tower_2 = Conv2D(256, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(y)
tower_3 = Conv2D(128, (1, 1), padding='same', activation='relu')(tower_3)

tower_4 = Conv2D(128, (1, 1), padding='same', activation='relu')(y)

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
    learning_rate=0.0005,
    rho=0.9,
    momentum=0.9,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)

# Initialize the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=1,          # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore the best weights when stopped
)

# Initialize the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=2, verbose=1)

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
              callbacks=[print_callback, early_stopping],
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

# Predict classes using the test set
y_pred = model.predict(test_images, batch_size=16, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

# Convert one-hot encoded test_labels to single labels
y_test_single_label = np.argmax(test_labels, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_single_label, y_pred_bool))

# Save the model (optional)
# model.save("emnist_model.h5")
