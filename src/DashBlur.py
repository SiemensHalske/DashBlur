import gc
gc.collect()
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
import emnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback

# Custom callback


class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nEpoch {epoch+1} has finished! - Loss: {logs['loss']}, Accuracy: {logs['accuracy']}")


# Load MNIST dataset
train_images, train_labels = emnist.extract_training_samples('byclass')
test_images, test_labels = emnist.extract_test_samples('byclass')


print(train_images.shape)
print(test_images.shape)
print("\n\n\n\n")

# Preprocess the data
train_images = train_images.reshape((697932, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((116323, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and save the history
history = model.fit(train_images, train_labels, epochs=10,
                    batch_size=16, callbacks=[CustomCallback()])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Plotting the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()
