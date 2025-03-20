# Libraries
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import torch
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F # Contiene una versione funzionale di molti layer. 
import torchvision.transforms as transforms
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import torchvision
import random
import wandb

# Data import

def import_CIFAR():

    train_data = CIFAR10(root='./data', train=True, download=True, transform=None)
    mean = (train_data.data /255.0).mean()
    std =(train_data.data /255.0).std()

    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)) 
    ])

    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    eval_size = round(eval_percentage*(len(train_data)))
    eval_data = Subset(train_data, range(eval_size))
    train_data = Subset(train_data, range(eval_size, len(train_data)))
    test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
    evaluation   = torch.utils.data.DataLoader(eval_data, batch_size, num_workers=4)
    test  = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, num_workers=4)

    
    return train, evaluation, test

    
def evaluate(model, dataset, device='cpu', test=False):
    model.eval()
    predictions = []
    ground_truth = [] 
    for (value, label) in tqdm(dataset, desc='Evaluating', leave=True):
        value = value.to(device)
        prediction = torch.argmax(model(value), dim=1)

        ground_truth = np.append(ground_truth, label.cpu().numpy())  
        predictions = np.append(predictions, prediction.detach().cpu().numpy()) 


    accuracy = accuracy_score(ground_truth, predictions)
    report = classification_report(ground_truth, predictions, zero_division=0, digits=3)
        
    
    return (accuracy, report)

# Simple function to plot the loss curve and validation accuracy.
def plot_validation_curves(losses_and_accs):
    losses = [x for (x, _) in losses_and_accs]
    accs = [x for (_, x) in losses_and_accs]
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss per Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(accs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Best Accuracy = {np.max(accs)} @ epoch {np.argmax(accs)}')

def train_batch(epoch, model, optimizer, loss_fn, train_data, device, writer):
    losses = []
    for (value, label) in tqdm(train_data, desc=f'Training epoch {epoch}', leave=True):
        value = value.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        prediction = model(value)
        
        # Choose loss function
        if loss == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()
            
        loss_value = loss_fn(prediction, label)
        loss_value.backward()
        optimizer.step()
        losses.append(loss_value.item())

        writer.add_scalar('Loss/train', np.mean(losses), epoch)
        
    return np.mean(losses)

def train_model(model, optimizer, loss, epochs, train_data, eval_data, device, writer, model_type = 'unknown_model'):
    model.train()
    losses_and_accs = []
    for epoch in range(epochs):
        loss_value = train_batch(epoch, model, optimizer, loss, train_data, device, writer)
        (accuracy, _) = evaluate(model, eval_data, device)
        losses_and_accs.append((loss_value, accuracy))
        #losses_and_accs.append(loss_value)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        #run.log({"acc": accuracy, "loss": loss_value})

        if epoch%5 == 0:
            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss_and_accs': losses_and_accs}
            
            torch.save(checkpoint, f"trained_models/model={model_type}-Epoch={epoch}-lr={lr}-opt={opt}-loss={loss}-epochs={epochs}-batch_size={batch_size}.pth")
    
    return losses_and_accs

    
# Models

class skipMLPBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, out_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(out_size, in_size)

    def forward(self, x):
        out1 = self.layer1(x)
        out1_relu = self.relu1(out1)
        out2 = self.layer2(out1_relu)

        out = x + out2
            
        return out

class skipMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(f'layer-flatten', nn.Flatten())
        
        for i in range(len(layer_sizes)-1):
            if i == 0:
                self.layers.add_module(f'layer-{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            elif i == len(layer_sizes)-2:
                self.layers.add_module(f'layer-{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            else:
                self.layers.add_module(f'layer-{i}', skipMLPBlock(layer_sizes[i], layer_sizes[i+1]))
                if layer_sizes[i] != layer_sizes[i+1]:
                    self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())
                    self.layers.add_module(f'layer-bridge{i}-{i+1}', nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                    self.layers.add_module(f'Nonlinearity-bridge{i}-{i+1}', nn.ReLU())
            if i < len(layer_sizes)-2:
                    self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())

    def forward(self, x):
        return self.layers(x)



class simpleCONV(nn.Module):
    def __init__(self, layer_sizes, size=28, in_channels=1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        countMaxPools = 0
        self.layers = nn.Sequential()
        current_size = size
  
        for i in range(len(layer_sizes)-2):
            self.layers.add_module(f'layer-{i}', nn.Conv2d(in_channels, layer_sizes[i], kernel_size, stride, padding))
            in_channels = layer_sizes[i]
            self.layers.add_module(f'batch_norm-{i}', nn.BatchNorm2d(layer_sizes[i]))
            self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())
            current_size = int((current_size - kernel_size + 2 * padding) / (stride))+1
            self.layers.add_module(f'MaxPool-{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            current_size = int(current_size/2)

        self.layers.add_module(f'layer-flatten', nn.Flatten())
        self.layers.add_module(f'linear-layer', nn.Linear(layer_sizes[-2]*current_size*current_size, layer_sizes[-1]))
     
    def forward(self, x):
        return self.layers(x)


def main():
    # Hyperparameters
    eval_percentage = 0.3
    lr = 0.0001
    opt = 'Adam'
    loss = 'CrossEntropy'
    epochs = 30
    #layers = [28*28, 64, 64, 64, 10]
    #layers = [28*28, 64, 128, 64, 64, 10] #MNIST sizes
    layers = [32*32*3, 64, 128, 64, 64, 10] #CIFAR10 sizes
    batch_size = 128
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="luca-cicchese-universit-di-firenze",
        # Set the wandb project where this run will be logged.
        project="AWS",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "skipMLP",
            "dataset": "CIFAR10",
            "epochs": epochs,
        },
    )

    # Main function skip connections
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    print("Loading dataset...")
    (train_data, eval_data, test_data) = import_CIFAR()
    
    # Instance of the model
    print(f"Training a skip connections MLP with {opt} and {loss}")
    modelskip = skipMLP(layers)
    modelskip.to(device)
    #print(model)
    
    # Choose optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(modelskip.parameters(), lr=lr)
        
    writer = SummaryWriter(log_dir=f"runs/MNIST-model='skipMLP'-lr={lr}-opt={opt}-loss={loss}-epochs={epochs}-batch_size={batch_size}-layers={layers}")
    
    losses_and_accs = train_model(modelskip, optimizer, loss, epochs, train_data, eval_data, device, writer)
    
    print(f"Minimum loss = {np.min(losses_and_accs)}")
    (accuracy, _) = evaluate(model, test_data, device=device)
    
    print(f"Accuracy on test set: {accuracy}")
    
    # Main function CNN 
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="luca-cicchese-universit-di-firenze",
        # Set the wandb project where this run will be logged.
        project="CNN_on_AWS",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": lr,
            "architecture": "simpleCNN",
            "dataset": "CIFAR10",
            "epochs": epochs,
        },
    )
    # Instance of the model
    print(f"Training a simple CNN with {opt} and {loss}")
    model = simpleCONV(layers)
    model.to(device)
    #print(model)
    
    # Choose optimizer
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    writer = SummaryWriter(log_dir=f"runs/CIFAR10-model='simpleCONV'-lr={lr}-opt={opt}-loss={loss}-epochs={epochs}-batch_size={batch_size}-layers={layers}")
    
    losses_and_accs = train_model(model, optimizer, loss, epochs, train_data, eval_data, device, writer)
    
    print(f"Minimum loss = {np.min(losses_and_accs)}")
    (accuracy, _) = evaluate(model, test_data, device=device)
    
    print(f"Accuracy on test set for CNN: {accuracy}")
    

    
if __name__ == "__main__":
    main()