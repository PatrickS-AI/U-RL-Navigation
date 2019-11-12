import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """ DQN class which is inherited from the PyTorch Model"""

    def __init__(self, state_size, action_size, seed, hidden_units =64):
        """Builds PyTorch neural network model according to input..

           Parameters
           =========
           state_size(int): Integer value describing the number of states
           action_size(int): Integer value describing the number of actions
           seed(float): Random seed value
           hidden_units(int): Number of hidden unints in NN layers
        """

        ## Calling from inherited class
        super(DQN, self).__init__()
        # Setting random seed from input
        self.seed = torch.manual_seed(seed)
        # Building NN with 3 layers
        self.fc1 = nn.Linear(state_size,hidden_units)
        self.fc2 = nn.Linear(hidden_units,hidden_units)
        self.fc3 = nn.Linear(hidden_units,action_size)


    def forward(self, state):
        """Defines the value propagation of values in the network.

        Parameters
        =========
        state(tensor): an input state of the environment
        """
        # Architecture 3 layers connected with 2 Relu layers
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
