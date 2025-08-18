from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import numpy as np

def extract_features(model, dataloader, device='cpu'):
    features = []
    labels = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            # Get CLS token embedding
            cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)

            # Append features and labels
            features.append(cls_output.cpu().numpy())
            labels.append(batch['label'].cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

config ={

    "dataset_name": "rotten_tomatoes",
    "model_name": "distilbert-base-uncased",
    "model_max_length": 512,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,
}

dataset_name = "rotten_tomatoes"
model_name = "distilbert-base-uncased"

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

tokenized_training = training_split.map( lambda example: tokenizer(example['text'], padding='max_length', truncation=True, max_length=config['model_max_length']), batched=True)
tokenized_validation = validation_split.map( lambda example: tokenizer(example['text'], padding='max_length', truncation=True, max_length=config['model_max_length']), batched=True)
tokenized_test = test_split.map( lambda example: tokenizer(example['text'], padding='max_length', truncation=True, max_length=config['model_max_length']), batched=True)

columns = ['input_ids', 'attention_mask', 'label']
tokenized_training.set_format(type='torch', columns=columns)
tokenized_validation.set_format(type='torch', columns=columns)
tokenized_test.set_format(type='torch', columns=columns)

train_loader = DataLoader(tokenized_training, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(tokenized_validation, batch_size=config['batch_size'], shuffle=False)
test_loader = DataLoader(tokenized_test, batch_size=config['batch_size'], shuffle=False)



train_features, train_labels = extract_features(model, train_loader, device)

model.eval()
validation_features, validation_labels = extract_features(model, val_loader, device)
test_features, test_labels = extract_features(model, test_loader, device)

print(f"Training classifier on extracted features...")
classifier = SVC()
classifier.fit(train_features, train_labels)

print("Classifier training complete.")
print("Evaluating classifier...")
validation_predictions = classifier.predict(validation_features)
accuracy = classifier.score(validation_features, validation_predictions)
print(f"Val accuracy: {accuracy:.2f}")

test_predictions = classifier.predict(test_features)
accuracy = classifier.score(test_features, test_predictions)
print(f"Test accuracy: {accuracy:.2f}")