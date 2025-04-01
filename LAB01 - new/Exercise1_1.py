# Libraries
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import torchvision
import random
import wandb

from dataset import load_dataset
from training import train_model
from evaluate import evaluate
import models



if __name__ == "__main__":

    # Hyperparameters
    eval_percentage = 0.3
    lr = 0.0001
    opt = 'Adam'
    loss = 'CrossEntropy'
    epochs = 5
    layers = [28*28, 64, 128, 64, 64, 10] 
    batch_size = 64
    dataset_name = 'MNIST'


    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="luca-cicchese-universit-di-firenze",
        # Set the wandb project where this run will be logged.
        project="CNN_on_AWS",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "simpleCNN",
            "dataset": "MNIST",
            "epochs": epochs,
        },
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    print("Loading dataset...")
    (train_data, eval_data, test_data) = load_dataset(dataset_name)


    # Instance of the model
    print(f"Training a simple MPL with {opt} and {loss}")
    model = models.simpleMLP(layers)
    model.to(device)
    #print(model)

    # Choose optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=f"runs/MNIST-model='MLP'-lr={lr}-opt={opt}-loss={loss}-epochs={epochs}-batch_size={batch_size}-layers={layers}")

    losses_and_accs = train_model(model, optimizer, loss, epochs, train_data, eval_data, device, writer)

    print(f"Minimum loss = {np.min(losses_and_accs)}")
    accuracy = evaluate(model, test_data, device=device)

    print(f"Accuracy on test set: {accuracy}")