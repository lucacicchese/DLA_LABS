"""
Training function for deep learning models using pytorch

The module includes the following functions:
    - init_logging(config)
    - log_epoch(epoch, writer, val_loss, val_accuracy)
    - save_checkpoint(checkpoint, checkpoint_dir)
    - get_optimizer(model, config)
    - get_loss(config)
    - train_model(model, train_loader, val_loader, config, device)

Author:
    Luca G. Cicchese
"""

import os
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

from evaluate import evaluate


def init_logging(config):
    """
    Initialize environment by:
    - creating necessary folders 
    - starting tensorboard and wandb
    
    Args:
    	config (dict): dictionary containing all hyperparameters
        
    Returns:
        SummaryWriter: tensorboard writer

    """

    # Setup environment
    directories = [
        f"logs/{config['logging']['tb_logs']}",
        f"logs/{config['logging']['save_dir']}",
    ]

    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Configure wandb
    if config["logging"]["weightsandbiases"]:
        wandb.init(
            project=config["project_name"],
            config=config,
            dir = "/logs/"
        )

    # Configure TensorBoard
    if config["logging"]["tensorboard"]:
        writer = SummaryWriter(
            log_dir=f"logs/{config['logging']['tb_logs']}/{config['training']}{timestamp}",
        )

        print(f"Saving logs to: logs/{config['logging']['tb_logs']}")


        return writer

    return None


def log_epoch(epoch, train_loss, val_loss, val_accuracy, config, writer = None):
    """
    Log training and validation loss and accuracy in tensorboard and wandb

    Args:
    	epoch (int): Current epoch
        val_loss (float): loss on validation set
        val_accuracy (float): accuracy on valiadtion set
        config (dict): dictionary containing all hyperparameters
    	writer (SummaryWriter): tensorboard SummaryWriter
        
    Returns:	
        
    """
    if config['logging']['wandb']:
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, step=epoch)
        # Logging

    if config['logging']['tensorboard']:
        writer.add_scalar("Loss/Training", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)


def save_checkpoint(checkpoint, checkpoint_dir):
    """
    Saves checkpoint in chosen location
    
    Args:
    	checkpoint (dict): All parameters to
    	checkpoint_dir (str): location to save checkpoints
        
    Returns:

    """
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"{checkpoint['model']}_epoch_{checkpoint['epoch']}.pth"))
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"{checkpoint['model']}_latest.pth"))


def get_optimizer(model, config):
    """
    Get optimizer based on config.

    Args:
        model (nn.Module): PyTorch model
        config (dict): Configuration dictionary

    Returns:
        torch.optim.Optimizer: Selected optimizer
    """
    lr = config["training"]["learning_rate"]
    optimizer = config["training"]["optimizer"]

    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        opt = optim.Adam(model.parameters(), lr=lr)

    return opt


def get_loss(config):
    """
    Get loss function based on config.

    Args:
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Selected loss function
    """
    loss_name = str(config["training"]["loss_function"])

    if loss_name == "MSE":
        loss = nn.MSELoss()
    elif loss_name == "crossentropy":
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.MSELoss()
    return loss


def train_model(model, train_loader, val_loader, config, device):
    """
    Train deep learning model with comprehensive tracking and checkpointing.

    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Configuration dictionary
        device (torch.device): Device to train on
        resume_training (bool): Whether to resume from existing checkpoint

    Returns:
        tuple: Training and validation losses and accuracies
    """

    tb_writer = init_logging(config)
    

    # Set optimizer and loss
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss(config)

    # Training hyperparameters
    epochs = config["training"]["epochs"]
    save_interval = config["logging"]["save_frequency"]

    # Checkpoint handling
    start_epoch = 0
    checkpoint_dir = f"logs/{config['logging']['save_dir']}"
    val_losses = []
    val_accuracies = []

    if config['training']['resume']:
        checkpoint_path = os.path.join(checkpoint_dir, f"{config['model']['type']}_latest.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            val_losses = checkpoint["val_losses"]
            val_accuracies = checkpoint["val_accuracies"]
            print(f"Resuming training from epoch {start_epoch}")

    # Move model to device
    print(f"val_losses: {val_losses}")
    print(f"val_accuracies: {val_accuracies}")


    model.to(device)


    # Main training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        train_loss = 0

        for (inputs, targets) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loss = total_train_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        # Tracking
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        # Log to wandb and tensorboard
        log_epoch(epoch, train_loss, val_loss, val_accuracy, config, tb_writer)

        

        # Checkpointing
        if (epoch + 1) % save_interval == 0  or epoch == epochs-1:
            checkpoint = {
                "model": config['model']['type'],
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
            }
            save_checkpoint(checkpoint, checkpoint_dir)

    return (val_losses, val_accuracies)
