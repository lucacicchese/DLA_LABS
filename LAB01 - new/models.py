import torch.nn as nn

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
    

