from torch.distributions import MultivariateNormal
import torch.nn as nn
import torch
import numpy as np


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
        
    def forward(self, x):
        value = self.value_function(x)
        return value
    

class StateDist():
    def __init__(self, dim_state, n_components, means, sigma):
        super(StateDist, self).__init__()
        self.n_components = n_components
        self.dim_state = dim_state
        self.means = means
        self.logstd = torch.ones(n_components, dim_state)*torch.diag(sigma).log()    # when using different diagonal matrices    
     
    def evaluate_state(self, state):
        s_log_probs = []
        for component in range(self.n_components):
            dist = self.evaluate(component)
            s_log_probs.append(dist.log_prob(state))
        component = np.argmax(s_log_probs)
        return component, s_log_probs[component]

    def evaluate(self, component):
        mean = self.means[component]
        std = self.logstd[component].exp()
        dist = MultivariateNormal(mean, torch.diag(std))        
        return dist
    
    def sample(self, component):
        state = self.evaluate(component).sample()
        return state

    
class PhaseDist():
    def __init__(self, n_components):
        self.n_components = n_components
        self.values = np.linspace(0, 1, n_components)
        self.positions = []
        
    def fit(self, n_frames):
        split = np.linspace(0, n_frames-1, self.n_components)
        self.positions = [round(x) for x in split]        
        return self.values, self.positions

    def sample(self):
        index = np.random.randint(0, self.n_components)
        return index, self.values[index], self.positions[index]
    
    def log_proba(self):
        return np.log(1/self.n_components)