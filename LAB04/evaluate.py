"""
Evalute function for deep learning models using pytorch

The module includes the following functions:
    - evaluate(model, data_loader, loss_fn, device)

Author:
    Luca G. Cicchese
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def evaluate(model, data_loader, loss_fn, device):
    """
    Evaluate model performance on a given dataset.

    Args:
        model (nn.Module): PyTorch model to evaluate
        data_loader (torch.utils.data.DataLoader): data to evaluate validation/training set
        loss_fn (nn.Module): Loss function
        device (torch.device): Device to perform evaluation on

    Returns:
        float: Accuracy of the model on dataset
    """
    model.eval()
    predictions = []
    ground_truth = [] 
    total_loss = 0
    loss = 0
    total_samples = 0

    for (value, label) in tqdm(data_loader, desc='Evaluating', leave=True):
        value = value.to(device)

        output = model(value)
        loss = loss_fn(output, label)

        total_loss += loss.item() 
        total_samples += value.size(0)

        prediction = torch.argmax(model(value), dim=1)

        ground_truth = np.append(ground_truth, label.cpu().numpy())  
        predictions = np.append(predictions, prediction.detach().cpu().numpy()) 

    
    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(ground_truth, predictions)
        
    
    return avg_loss, accuracy

