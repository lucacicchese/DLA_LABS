import torch.nn as nn
import torch

class simpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) model
    """
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(f'layer-flatten', nn.Flatten())
        for i in range(len(layer_sizes)-1):
            self.layers.add_module(f'layer-{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())

    def forward(self, x):
        return self.layers(x)
    

class skipMLPBlock(nn.Module):
    """
    Skip connection block for MLP
    """
    def __init__(self, in_size, out_size):
        super().__init__()

        self.layer1 = nn.Linear(in_size, out_size)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(out_size, out_size)

        if in_size == out_size:
            self.bridge = nn.Identity()  
        else:
            self.bridge = nn.Linear(in_size, out_size)
            


    def forward(self, x):
        
        out1 = self.layer1(x)
        out1_relu = self.relu1(out1)
        out1_drop = self.dropout(out1_relu)
        out2 = self.layer2(out1_drop)
        x = self.bridge(x)

        out = x + out2
            
        return out

class skipMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model with skip connections
    """
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(f'layer-flatten', nn.Flatten())
        
        for i in range(len(layer_sizes)-1):
            # Add just the linear layer for the first and last layers
            if i == 0 or i == len(layer_sizes)-2:
                self.layers.add_module(f'layer-{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # For each middle layer add a skip connection block
            else:
                self.layers.add_module(f'layer-{i}', skipMLPBlock(layer_sizes[i], layer_sizes[i+1]))

            if i < len(layer_sizes)-2:
                    self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class my_CNN(nn.Module):
    def __init__(self, layer_sizes, size=28, in_channels=1, kernel_size=3, stride=1, padding=1, num_classes=10):
        super().__init__()
        countMaxPools = 0
        self.layers = nn.Sequential()
        current_size = size

        self.activations = []
  
        for i in range(len(layer_sizes)-1):
            self.layers.add_module(f'conv-layer-{i}', nn.Conv2d(in_channels, layer_sizes[i], kernel_size, stride, padding))
            in_channels = layer_sizes[i]
            self.layers.add_module(f'batch_norm-{i}', nn.BatchNorm2d(layer_sizes[i]))
            self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())
            self.layers.add_module(f'Dropout-{i}', nn.Dropout2d(p=0.3))
            current_size = int((current_size - kernel_size + 2 * padding) / (stride))+1
            self.layers.add_module(f'MaxPool-{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            current_size = int(current_size/2)

        #self.last_conv = nn.Conv2d(layer_sizes[-2], layer_sizes[-1], kernel_size, stride, padding)
        self.last_conv = nn.Conv2d(in_channels, num_classes, kernel_size, stride, padding)
        self.last_relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        #self.linear_layer = nn.Linear(layer_sizes[-1], num_classes)

        #self._hook = self.layers[-5].register_forward_hook(self.save_activation)

    def forward(self, x):
        x = self.layers(x)   # up to GAP
        x = self.last_conv(x)
        x = self.last_relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        #x = self.linear_layer(x)
        
        return x

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def get_activation(self):
        return self.activations[-1]