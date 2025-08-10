import torch.nn as nn

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
        self.layers.add_module(f'softmax', nn.Softmax(dim=1))

        self._hook = self.layers[-5].register_forward_hook(self.save_activation)

    def forward(self, x):
        return self.layers(x)
    
    def save_activation(self, module, input, output):
        self.activations.append(output)

    def get_activation(self):
        return self.activations[-1]