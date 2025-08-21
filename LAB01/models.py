import torch.nn as nn
import torch

class simpleMLP(nn.Module):
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
        print(f"Input size: {current_size}")
  
        for i in range(len(layer_sizes)-2):
            self.layers.add_module(f'layer-{i}', nn.Conv2d(in_channels, layer_sizes[i], kernel_size, stride, padding))
            in_channels = layer_sizes[i]
            self.layers.add_module(f'batch_norm-{i}', nn.BatchNorm2d(layer_sizes[i]))
            self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())
            current_size = int((current_size - kernel_size + 2 * padding) / (stride))+1
            self.layers.add_module(f'MaxPool-{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            current_size = int(current_size/2)
            print(f"Current size after layer {i}: {current_size}")

        self.layers.add_module(f'layer-flatten', nn.Flatten())
        self.layers.add_module(f'linear-layer', nn.Linear(layer_sizes[-2]*current_size*current_size, layer_sizes[-1]))
     
    def forward(self, x):
        return self.layers(x)


class my_CNN(nn.Module):
    def __init__(self, layer_sizes, size=28, in_channels=1, kernel_size=3, stride=1, padding=1, num_classes=10):
        super().__init__()
        countMaxPools = 0
        self.layers = nn.Sequential()
        current_size = size

        self.activations = []
  
        for i in range(len(layer_sizes)-2):
            self.layers.add_module(f'conv-layer-{i}', nn.Conv2d(in_channels, layer_sizes[i], kernel_size, stride, padding))
            in_channels = layer_sizes[i]
            self.layers.add_module(f'batch_norm-{i}', nn.BatchNorm2d(layer_sizes[i]))
            self.layers.add_module(f'Nonlinearity-layer-{i}', nn.ReLU())
            current_size = int((current_size - kernel_size + 2 * padding) / (stride))+1
            self.layers.add_module(f'MaxPool-{i}', nn.MaxPool2d(kernel_size=2, stride=2))
            current_size = int(current_size/2)

        self.layers.add_module(f'last-conv', nn.Conv2d(layer_sizes[-2], layer_sizes[-1], kernel_size, stride, padding))
        self.layers.add_module(f'last-relu', nn.ReLU())
        self.layers.add_module(f'GAP', nn.AdaptiveAvgPool2d(1))
        self.layers.add_module(f'linear-layer', nn.Linear(layer_sizes[-1], num_classes))


        self._hook = self.layers[-5].register_forward_hook(self.save_activation)

    def forward(self, x):
        x = self.layers[:-3](x)   # up to GAP
        print("Before GAP:", x.shape)
        x = self.layers[-3](x)    # GAP
        print("After GAP:", x.shape)
        x = torch.flatten(x, 1)
        x = self.layers[-2](x)    # Linear
        return self.layers[-1](x)
    
    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def get_activation(self):
        return self.activations[-1]