import json
import random

def split_dataset(input_file, train_file, eval_file, split_ratio=0.8):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Calculate the number of lines for training
    num_train = int(len(lines) * split_ratio)

    # Split the lines into training and evaluation sets
    train_lines = lines[:num_train]
    eval_lines = lines[num_train:]

    # Write the training lines to the training file
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # Write the evaluation lines to the evaluation file
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)

# Load data from JSON file
with open('german_nlp_dataset.json', 'r') as f:
    data = json.load(f)

# Split the data into training and evaluation sets
split_dataset('german_nlp_dataset.json', 'train_dataset.txt', 'eval_dataset.txt')
