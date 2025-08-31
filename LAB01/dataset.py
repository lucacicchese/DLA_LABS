"""
Function that loads the desired dataset

The module includes the following functions:
    - simple_transformation(mean, std)
    - load_dataset(dataset_name, val_percentage, batch_size)

Author:
    Luca G. Cicchese
"""

import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.datasets as datasets

    
def simple_transformation(mean, std, dataset='cifar10'):
    """
    This function creates the standard transformation of data:
    - converts data to tensor
    - normalizes data
    
    Args:
    	mean (float): maen value in the dataset
        std (float): standard deviation in the dataset
        
    Returns:
    	
    """

    if dataset.lower() == 'cifar10':
        
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def load_dataset(dataset_name='MNIST', val_percentage = 0.3, batch_size = 64):
    """
        This function loads the desired dataset
        
        Args:
            dataset_name (str): name of the dataset to load
            val_percentage (float): percentage of training set to use for validation
            batch_size (int): size of the batches
            
        Returns:
            DataLoader: training set
            DataLoader: validation set
            DataLoader: testing set

    """

    if dataset_name.lower() == 'mnist':
        dataset = datasets.MNIST(root='./data', train=True, download = True)
        mean = (dataset.data /255.0).mean()
        std = (dataset.data /255.0).std()

        transform = simple_transformation(mean, std, dataset='mnist')

        ds_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
        ds_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform) 

    elif dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=True,  download = True)
        mean = (dataset.data /255.0).mean()
        std = (dataset.data /255.0).std()

        transform = simple_transformation(mean, std)

        ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
        ds_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) 


    train_size = int(len(ds_train)*(1-val_percentage))
    val_size = len(ds_train)-train_size
    ds_train, ds_val = random_split(ds_train, [train_size, val_size])


    train_loader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4)    
    

    return train_loader, val_loader, test_loader

