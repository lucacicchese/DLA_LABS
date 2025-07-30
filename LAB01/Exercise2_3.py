# Use the CNN model you trained in Exercise 1.3 and implement 
# [*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.):> B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. 
# Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015). 
# Use your CNN implementation to demonstrate how your trained CNN *attends* to specific image features 
# to recognize *specific* classes. Try your implementation out using a pre-trained ResNet-18 model and 
# some images from the 
# [Imagenette](https://pytorch.org/vision/0.20/generated/torchvision.datasets.Imagenette.html#torchvision.datasets.Imagenette) dataset 
# -- I suggest you start with the low resolution version of images at 160px.

# Libraries
from dataset import load_dataset
from training import train_model
from evaluate import evaluate
import models

import numpy as np
import torch



def class_activation_map(model, input_image, target_class):
    """
    Generate Class Activation Map for a given input image and target class.
    
    Args:
        model (torch.nn.Module): The trained CNN model.
        input_image (torch.Tensor): The input image tensor.
        target_class (int): The target class index for which to generate the CAM.
        
    Returns:
        torch.Tensor: The Class Activation Map.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_image.unsqueeze(0))
        output = output[0, target_class]
        
        # Get the gradients of the output with respect to the parameters of the model
        output.backward()
        
        # Get the gradients of the last convolutional layer
        gradients = model.get_last_conv_layer().weight.grad
        
        # Global Average Pooling
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Multiply each channel in the feature map by the corresponding gradient
        for i in range(len(pooled_gradients)):
            model.get_last_conv_layer().weight[i] *= pooled_gradients[i]
        
        # Generate the Class Activation Map
        cam = model.get_last_conv_layer().output.squeeze().cpu().numpy()
        
    return cam

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