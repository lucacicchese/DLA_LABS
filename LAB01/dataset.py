"""
Function that loads the desired dataset

The module includes the following functions:
    - simple_transformation(mean, std)
    - load_dataset(dataset_name, val_percentage, batch_size)

"""

# Import external libraries
import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import torchvision.datasets as datasets
import tarfile
import urllib.request

class ImagenetteLabels(Dataset):
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label
    
    def get_class_name(self, label):
        return self.class_names[label]


def download_and_load_imagenette(data_dir='data'):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    archive_path = os.path.join(data_dir, 'imagenette2.tgz')
    extracted_folder = os.path.join(data_dir, 'imagenette2')

    os.makedirs(data_dir, exist_ok=True)

    # Download if not already downloaded
    if not os.path.exists(archive_path):
        print(f"Downloading Imagenette dataset to {archive_path}...")
        urllib.request.urlretrieve(url, archive_path)
        print("Download complete.")

    # Extract if not already extracted
    if not os.path.exists(extracted_folder):
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    
    return
    
def simple_transformation(mean=0, std=1, dataset='cifar10'):
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
    elif dataset.lower() == 'imagenette':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
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


        transform = simple_transformation(dataset='cifar10')

        ds_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) 
        ds_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) 

    elif dataset_name.lower() == 'imagenette':
        download_and_load_imagenette(data_dir='./data')

        transform = simple_transformation(dataset='imagenette')

        lbl_dict = dict(
            n01440764='tench',
            n02102040='English springer',
            n02979186='cassette player',
            n03000684='chain saw',
            n03028079='church',
            n03394916='French horn',
            n03417042='garbage truck',
            n03425413='gas pump',
            n03445777='golf ball',
            n03888257='parachute'
        )

        ds_train = datasets.ImageFolder(
            root='./data/imagenette2/train',
            transform=transform
        )
        ds_test = datasets.ImageFolder(
            root='./data/imagenette2/val',
            transform=transform
        )

        class_names = []
        for cls in ds_train.classes:
            class_name = lbl_dict[cls]
            class_names.append(class_name)


        ds_train = ImagenetteLabels(ds_train, class_names)
        ds_test  = ImagenetteLabels(ds_test, class_names)

    train_size = int(len(ds_train)*(1-val_percentage))
    val_size = len(ds_train)-train_size
    ds_train, ds_val = random_split(ds_train, [train_size, val_size])


    train_loader = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True, num_workers=4)
    val_loader   = torch.utils.data.DataLoader(ds_val, batch_size, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(ds_test, batch_size, shuffle=True, num_workers=4)    
    

    return train_loader, val_loader, test_loader

