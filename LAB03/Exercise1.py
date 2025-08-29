from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import extract_features


if __name__ == "__main__":
    config ={
        "project_name": "LAB03_Exercise1",
        "dataset_name": "rotten_tomatoes",
        "model_name": "distilbert-base-uncased",
        "model_max_length": 512,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 2e-5,

    "logging": {
        "tensorboard": True,
        "weightsandbiases": True,
        "wandb": True,  
        "tb_logs": "tensorboard_runs",  
        "save_dir": "checkpoints",      
        "save_frequency": 1             
    }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config["logging"]["wandb"]: 
        wandb.init(project="svm-text-classifier", config=config)
    if config["logging"]["tensorboard"]:
        writer = SummaryWriter(log_dir=f"logs/{config['logging']['tb_logs']}")

    feature_extractor = pipeline("feature-extraction", model=config['model_name'], framework="pt")

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


    train_features, train_labels = extract_features(model, feature_extractor, training_split, config, device)
    validation_features, validation_labels = extract_features(model, feature_extractor, validation_split, config, device)
    test_features, test_labels = extract_features(model, feature_extractor, test_split, config, device)

    print("Train feature shape:", train_features.shape)
    print("Validation feature shape:", validation_features.shape)
    print("Test feature shape:", test_features.shape)


    print(f"Starting training classifier on extracted features")
    classifier = SVC()
    classifier.fit(train_features, train_labels)


    print("Classifier training complete.")
    print("Evaluating classifier...")
    validation_predictions = classifier.predict(validation_features)
    print("Validation_predictions shape:", validation_predictions.shape)
    print("Validation_labels shape:", validation_labels.shape)
    print(classification_report(validation_labels, validation_predictions))

    val_acc = accuracy_score(validation_labels, validation_predictions)
    print(f"Validation Accuracy: {val_acc:.2f}")

    if config["logging"]["wandb"]:
        wandb.log({"val_accuracy": val_acc})
    if config["logging"]["tensorboard"]:
        writer.add_scalar("Accuracy/Validation", val_acc, 0)


    print("Evaluating on test set...")
    test_predictions = classifier.predict(test_features)
    print("Test_predictions shape:", test_predictions.shape)
    print("Test_labels shape:", test_labels.shape)
    print(classification_report(test_labels, test_predictions))

    test_acc = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_acc:.2f}")

    if config["logging"]["wandb"]:
        wandb.log({"test_accuracy": test_acc})
    if config["logging"]["tensorboard"]:
        writer.add_scalar("Accuracy/Test", test_acc, 0)
