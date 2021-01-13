from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -0.1, 0.1)
        nn.init.uniform_(m.bias, -0.1, 0.1)

class PNet(nn.Module):
    def __init__(self, dim_state, dim_action, scale=0.01):
        super(PNet, self).__init__()
         
        self.policy_mean = nn.Sequential(
            nn.Linear(dim_state, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, dim_action)
        )
        
        self.sigma = scale*torch.eye(dim_action)
        self.apply(init_weights)

    def forward(self, x):
        mu = self.policy_mean(x)
        policy = MultivariateNormal(mu, self.sigma)
        return policy


class VNet(nn.Module):
    def __init__(self, dim_state):
        super(VNet, self).__init__()
        
        self.value_function = nn.Sequential(
            nn.Linear(dim_state, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.apply(init_weights)
        
    def forward(self, x):
        value = self.value_function(x)
        return value