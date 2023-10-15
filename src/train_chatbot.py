import json
import tensorflow as tf
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers import Embedding, Flatten
from keras.layers import LSTM, Bidirectional, BatchNormalization

L2_REG = 0.001


optimizer_adam = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)


def schedule(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
        # return lr * .9


def warm_restart(epoch, lr, cycle_length=50, lr_min=0.0001, lr_max=0.001):
    return lr_min + 0.7 * (lr_max - lr_min) * (1 + math.cos(math.pi * (epoch % cycle_length) / cycle_length))


def cosine_annealing(epoch, lr, T_max=100, eta_min=0.00001):
    return eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max))


def step_decay(epoch, lr):
    drop_rate = 0.9
    epochs_drop = 15.0
    return lr * (drop_rate ** np.floor((1+epoch)/epochs_drop))


# Load the dataset
with open('/home/hendrik/Documents/Projects/DashBlur/data/german_nlp_dataset.json') as file:
    dataset = json.load(file)

# Initialize lists for patterns and tags (labels)
patterns = []
tags = []

for intent in dataset['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Use LabelEncoder to convert tags to numbers
label_encoder = LabelEncoder()
y_data = label_encoder.fit_transform(tags)

# One-hot encoding of labels
y_data = to_categorical(y_data)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_data = vectorizer.fit_transform(patterns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.4, random_state=42)

X_train = X_train.toarray()
X_test = X_test.toarray()

print('\n\n\n')
print('X_train shape: %s' % str(X_train.shape))
print('y_train shape: %s' % str(y_train.shape))
print('X_test shape: %s' % str(X_test.shape))
print('y_test shape: %s' % str(y_test.shape))
print('\n\n\n')

VOCAB_SIZE = 1000
EMBEDDING_DIM = 100
MAX_LENGTH = 526
NUM_CLASSES = 76

# model = Sequential([
#     Dense(512, input_shape=(
#        X_train.shape[1],), activation='relu', kernel_regularizer=l2(L2_REG)),
#     Dropout(0.2),
#     Dense(512, activation='relu', kernel_regularizer=l2(L2_REG)),
#     Dropout(0.2),
#     Dense(256, activation='relu', kernel_regularizer=l2(L2_REG)),
#     Dropout(0.2),
#     Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
#     Dropout(0.2),
#     Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
#     Dense(64, activation='softmax')
# ])

model = Sequential([
    Dense(512, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(512, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(256, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    
    Dense(76, activation='softmax')
])

model_2 = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])


# Initialize the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation accuracy
    patience=5,          # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True  # Restore the best weights when stopped
)

# Initialize the LearningRateScheduler callbacks
callback_warm_restart = LearningRateScheduler(lambda epoch: warm_restart(epoch, lr=0.001))
callback_cosine_annealing = LearningRateScheduler(lambda epoch: cosine_annealing(epoch, lr=0.001))
callback_step_decay = LearningRateScheduler(lambda epoch: step_decay(epoch, lr=0.001))
callback_schedule = LearningRateScheduler(schedule, verbose=1)

model.compile(
    optimizer=optimizer_adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=1,
    validation_split=0.3
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Loss: %s' % loss)
print('Accuracy: %s' % accuracy)
