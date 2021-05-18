import torch
import torch.nn as nn
import torch.optim as optim

def weight_init(layers, init = 'xavier'):
    for layer in layers:
        if hasattr(layer, 'weight'):
            if init == 'xavier':
                nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain('relu'))
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def Block(in_dim, out_dim, dropout=0.0, batch_norm = False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        # nn.Dropout(dropout) if not batch_norm else nn.BatchNorm1d(dim),
    )

class MLP(nn.Module):

    def __init__(self, state_size, action_size, layer_size, depth=0):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
                Block(self.state_size, layer_size),
                *[r for r in Block(layer_size, layer_size) for _ in range(depth)],
                nn.Linear(layer_size, action_size)
        )

        weight_init(self.model)
    
    def forward(self, state):
        return self.model(state)
