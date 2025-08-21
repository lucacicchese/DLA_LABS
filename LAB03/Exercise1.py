from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

def extract_features(model, feature_extractor, dataset, device='cpu'):
    features = []
    labels = []

    for i, example in enumerate(dataset):
        text = example['text']
        label = example['label']

        output_features = feature_extractor(text)
        cls_token_vector = output_features[0][0]  # Get the [CLS] token vector
        features.append(cls_token_vector)
        labels.append(label)

    return np.array(features), np.array(labels)



config ={

    "dataset_name": "rotten_tomatoes",
    "model_name": "distilbert-base-uncased",
    "model_max_length": 512,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = "rotten_tomatoes"
model_name = "distilbert-base-uncased"
feature_extractor = pipeline("feature-extraction", model=model_name, framework="pt")

# Exercise 1.1
print(f"Loading dataset: {config['dataset_name']}")
dataset = load_dataset(config['dataset_name'])
split_names = get_dataset_split_names(config['dataset_name'])

print(f"Available splits for {config['dataset_name']}: {split_names}")

# Exercise 1.2
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModel.from_pretrained(config['model_name'])

# Exercise 1.3
training_split = dataset['train']
validation_split = dataset['validation']
test_split = dataset['test']
print(f"Training split size: {len(training_split)}, Validation split size: {len(validation_split)}, Test split size: {len(test_split)}")


train_features, train_labels = extract_features(model, feature_extractor, training_split, device)
validation_features, validation_labels = extract_features(model, feature_extractor, validation_split, device)
test_features, test_labels = extract_features(model, feature_extractor, test_split, device)

print(f"Starting training classifier on extracted features")
classifier = SVC()
classifier.fit(train_features, train_labels)

print("Classifier training complete.")
print("Evaluating classifier...")
validation_predictions = classifier.predict(validation_features)
validation_predictions = validation_predictions.reshape(-1, 1)
validation_labels = validation_labels.reshape(-1, 1)
print("Validation_predictions shape:", validation_predictions.shape)
print("Validation_labels shape:", validation_labels.shape)
accuracy = classifier.score(validation_predictions, validation_labels)
print(f"Validation accuracy: {accuracy:.2f}")
print("Evaluating on test set...")

test_predictions = classifier.predict(test_features)
test_predictions = test_predictions.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)
print("Test_predictions shape:", test_predictions.shape)
print("Test_labels shape:", test_labels.shape)
accuracy = classifier.score(test_predictions, test_labels)
print(f"Test accuracy: {accuracy:.2f}")


