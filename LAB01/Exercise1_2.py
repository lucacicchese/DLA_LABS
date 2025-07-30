# Libraries
from dataset import load_dataset
from training import train_model, get_loss
from evaluate import evaluate
import models

import numpy as np
import torch


if __name__ == "__main__":

    config = {
    "project_name": "LAB01_Exercise1_2",  

    "dataset_name": "CIFAR10", 

    "training": {
        "eval_percentage": 0.3,
        "learning_rate": 0.0001,
        "optimizer": "adam", 
        "epochs": 20,
        "batch_size": 64,
        "resume": True, 
        "layers": [32*32*3, 64, 128, 64, 64, 10],
        "dataset_name": 'cifar10',
        "loss_function": "crossentropy"
    },

    "model": {
        "type": "skip-mlp",  
        "layers": [32*32*3, 64, 128, 64, 64, 10]  
    },

    "logging": {
        "tensorboard": True,
        "weightsandbiases": True,
        "wandb": True, 
        "tb_logs": "tensorboard_runs",  
        "save_dir": "checkpoints",     
        "save_frequency": 4            
    }
}




    train_hyperparameters = config["training"]


    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    print("Loading dataset...")
    (train_data, eval_data, test_data) = load_dataset(train_hyperparameters['dataset_name'])


    # Instance of the model
    print(f"Training an MPL with skip connections using {train_hyperparameters['optimizer']} and {train_hyperparameters['loss_function']}")
    model = models.skipMLP(train_hyperparameters['layers'])
    print(f"Model architecture: {model}")
    model.to(device)


    # Choose optimizer
    if train_hyperparameters['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hyperparameters['learning_rate'])


    losses, accuracies = train_model(model, train_data, eval_data, config, device)

    print(f"Minimum loss = {np.min(losses)}")
    print(f"Maximum accuracy = {np.max(accuracies)}")

    loss_fn = get_loss(config)
    test_loss, test_accuracy = evaluate(model, test_data, loss_fn, device=device)

    print(f"Loss on test set: {test_loss}")
    print(f"Accuracy on test set: {test_accuracy}")