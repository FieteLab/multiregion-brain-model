import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F

from towertask.utils import set_seed

"""for running with MESH, REINFORCE, 3 actions, ==> used in train.py"""

class RNNPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=3, loc_pred_size=1, alpha=0.025):
        super(RNNPolicy, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.core_hidden = None
        self.alpha = alpha
        self.saved_log_probs = []
        self.rewards = []
        # self._initialize_weights()
    
    def _initialize_weights(self):
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, x):
        x = x.view(1, 1, -1)
        device = x.device
        if self.core_hidden is None:
            self.core_hidden = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
        else:
            # Ensure hidden state is on the same device as input
            self.core_hidden = self.core_hidden.to(device)
        if torch.isnan(x).any():
            breakpoint()
            raise ValueError("Input contains NaN values")

        out, core_hidden_predecay = self.rnn(x, self.core_hidden)
        if torch.isnan(out).any() or torch.isnan(core_hidden_predecay).any():
            breakpoint()
            raise ValueError("RNN output contains NaN values")
    
        # Create a new tensor for the updated hidden state
        new_hidden = []
        for i in range(self.rnn.num_layers):
            new_layer_hidden = (1 - self.alpha) * self.core_hidden[i] + self.alpha * core_hidden_predecay[i]
            new_hidden.append(new_layer_hidden)

        # Stack the new hidden layers together
        self.core_hidden = torch.stack(new_hidden)
        if torch.isnan(self.core_hidden).any():
            raise ValueError("New hidden state contains NaN values")

        action_probs = self.fc(out.squeeze(0))
        if torch.isnan(action_probs).any():
            breakpoint()
            raise ValueError(f"Encountered NaN values in probs: {out}")
        return torch.softmax(action_probs, dim=1), core_hidden_predecay

    def clip_gradients(self):
        for param in self.parameters():
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_(param, max_norm=1)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mlp_train_rule='supervised'):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()                         # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Second fully connected layer
        
        self.saved_log_probs = []
        self.rewards = []
        self.train_rule = mlp_train_rule

    def forward(self, x):
        x = x.view(1, 1, -1)
        out = self.fc1(x)     # Pass input through first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)   # Pass through second layer
        if not self.train_rule == 'supervised':
            out = F.softmax(out, dim=-1)  # Apply softmax to convert to probabilities
        return out
    
# --------------------------
# Training and Main Loop
# --------------------------
def select_action(policy, state, return_prob_dist=False):
    """
    Select an action using the policy network.

    Args:
        policy: The policy network.
        state (Tensor): Current state input.
        return_prob_dist (bool): If True, also return the probability distribution.

    Returns:
        action (int): Sampled action.
        hidden_state (Tensor): Hidden state output from the policy.
        probs (list, optional): Action probabilities (if return_prob_dist is True).
    """
    device = policy.fc.weight.device
    state = torch.FloatTensor(state.to(device))
    probs, hidden_state = policy(state)
    m = Categorical(probs)
    action = m.sample()

    policy.saved_log_probs.append(m.log_prob(action))

    if return_prob_dist:
        return action.item(), hidden_state, probs.squeeze().tolist()
    else:
        return action.item(), hidden_state

def finish_episode(policy, optimizer, gamma=0.99):
    """
    Policy Graident.
    """
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    
    policy_loss = torch.cat(policy_loss).sum()
    # print(f'\t loss is {policy_loss}')
    optimizer.zero_grad()
    policy_loss.backward()
    
    # Clip gradients
    policy.clip_gradients()
    
    optimizer.step()
    
    try:
        policy.core_hidden = policy.core_hidden.detach()
    except:
        policy.core_hidden = tuple(state.detach() for state in policy.core_hidden)
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def finish_MLP_episode(policy, optimizer, gamma=0.9): 
    R = 0
    policy_loss = []
    returns = deque()

    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    for log_prob, R in zip(policy.saved_log_probs, returns):
        log_prob = torch.tensor(log_prob).unsqueeze(0) if log_prob.dim() == 0 else log_prob
        if not log_prob.requires_grad:
            log_prob = log_prob.detach().requires_grad_()
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    # print(f'\t loss is {policy_loss}')
    optimizer.zero_grad()
    policy_loss.backward()
    
    # Clip gradients
    policy.clip_gradients()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.saved_log_probs[:]
