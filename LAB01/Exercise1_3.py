# Libraries
from dataset import load_dataset
from training import train_model
from evaluate import evaluate
import models

import numpy as np
import torch



if __name__ == "__main__":

    config = {
    "project_name": "deep_learning_project",  

    "dataset_name": "CIFAR10", 

    "training": {
        "eval_percentage": 0.3,
        "learning_rate": 0.0001,
        "optimizer": "adam", 
        "epochs": 5,
        "batch_size": 64,
        "resume": True, 
        "layers": [32*32*3, 64, 64, 10],
        "dataset_name": 'cifar10',
        "loss_function": "crossentropy"
    },

    "model": {
        "type": "simple_cnn",  
        "layers": [32*32*3, 64, 64, 10] 
    },

    "logging": {
        "tensorboard": False,
        "weightsandbiases": False,
        "wandb": False, 
        "tb_logs": "tensorboard_runs",  
        "save_dir": "checkpoints",     
        "save_frequency": 1            
    }
}


    train_hyperparameters = config["training"]


    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    print("Loading dataset...")
    (train_data, eval_data, test_data) = load_dataset(train_hyperparameters['dataset_name'])


    # Instance of the model
    print(f"Training an simple cnn using {train_hyperparameters['optimizer']} and {train_hyperparameters['loss_function']}")
    model = models.simpleCONV(train_hyperparameters['layers'], 32, 3)
    print(f"Model architecture: {model}")
    model.to(device)
    #print(model)

    # Choose optimizer
    if train_hyperparameters['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=train_hyperparameters['learning_rate'])

    accuracies = train_model(model, train_data, eval_data, config, device)

    print(f"Minimum loss = {np.min(accuracies)}")
    accuracy = evaluate(model, test_data, config["training"]["loss_function"], device=device)

    print(f"Accuracy on test set: {accuracy}")