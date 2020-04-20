import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, state):
        x = self.main(state)
        return x

