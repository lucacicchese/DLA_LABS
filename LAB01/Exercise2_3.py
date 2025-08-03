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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random



def class_activation_map(model, input_image, target_class):

    model.eval()
    model.zero_grad()
    output = model(input_image)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    output[:, class_idx].backward()

    activations = model.get_activation()

    gradients = torch.autograd.grad(outputs=output[:, class_idx], inputs=activations,
                                    grad_outputs=torch.ones_like(output[:, class_idx]), retain_graph=True)[0]
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Media su tutte le dimensioni spaziali
    cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Somma ponderata delle attivazioni

    cam = F.relu(cam)


    cam = cam - cam.min()
    cam = cam / cam.max()

    cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)

    cam = cam.squeeze().cpu().detach().numpy()


    input_image = input_image.squeeze().cpu().detach().numpy()   
    plt.imshow(input_image, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    config = {
    "project_name": "deep_learning_project",  

    "dataset_name": "Imagenette", 

    "training": {
        "eval_percentage": 0.3,
        "learning_rate": 0.0001,
        "optimizer": "adam", 
        "epochs": 5,
        "batch_size": 64,
        "resume": True, 
        "layers": [64, 64, 64, 10],
        "dataset_name": 'cifar10',
        "loss_function": "crossentropy"
    },

    "model": {
        "type": "simple_cnn",  
        "layers": [64, 64, 64, 10] 
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
    model = models.my_CNN(train_hyperparameters['layers'], 32, 3)
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

    num_samples = 15

    random_indices = random.sample(range(len(test_data)), num_samples)

    for idx in random_indices:
        image, label = test_data[idx]
        image = image.unsqueeze(0).to(device)
        
        class_activation_map(model, image, label.item())

    