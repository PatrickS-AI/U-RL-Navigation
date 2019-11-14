import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import DQN

import torch
import torch.nn.functional as F
import torch.optim as optim

# Select cpu or gpu as device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """
    RL Agent that interacts with a given environment, learns and adapts succesfull behaviour.
    """


    def __init__(self,state_size, action_size
                 ,batch_size,learn_step_size,buffer_size
                 ,gamma , learning_rate, tau
                 ,seed):
        """
        Intialize the agent and its learning parameter set.

        Parameters
        =========
        state_size (int): Size of the state space
        action_size (int): Size of the action space

        batch_size (int): Size of the batch size used in each learning step
        learn_step_size (int): Number of steps until agent ist trained again
        buffer_size (int): Size of replay memory buffer

        gamma (float): Discount rate that scales future discounts
        learning_rate (float): Learning rate of neural network
        tau (float): Update strenght between local and target network

        seed (float): Random set for initialization
        """

        # ----- Parameter init -----
        # State and action size from environment
        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer and learning properties
        self.batch_size      = batch_size
        self.learn_step_size = learn_step_size
        self.gamma = gamma
        self.tau  = tau

        # General
        self.seed = random.seed(seed)


        # ----- Network and memory init -----
        # Init identical NN as local and target networks and set optimizer
        self.qnetwork_local  = DQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Initialize replay memory and time step (for updating every learn_step_size steps)
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        """
        Append information of past step in memory and trigger learning.

        Parameters
        ==========
        state (array_like): State before action
        action (array_like): Action that was taken
        reward (float): Reward for action
        next_state (array_like): State after action
        done (bool): Indicator if env was solved after action
        """

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every learn_step_size time steps.
        self.t_step = (self.t_step + 1) % self.learn_step_size
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.get_memory_size()  > self.batch_size:
                self.learn()


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Parameters
        ==========
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Transform state to PyTorch tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Get action scores for state from network
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """
        Get sample of experience tuples and value parameters target network.
        """

        # Get tuples from experience buffer
        experiences = self.memory.get_sample()
        states, actions, rewards, next_states, dones = experiences

        #  -----DQN -----
        #Optional: to be replaced with Double DQN (see below)
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # ----- Double DQN -----
        # Detach to not update weights during learning
        # Select maximum value
        # Unsqueeze to reduce the tensor dimension to one
        expected_next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get Q values for next actions from target Q-network
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, expected_next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        # Gather values alon an axis specified by dim
        Q_expected = self.qnetwork_local(states).gather(1, actions)


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ----- Update target network -----
        #Soft update model parameters.
        #θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Creates Replay memmory for given buffer and batch size

        Parameters
        ==========
        action_size (int): Size of the state space
        buffer_size (int): Size of the action space
        batch_size (int): Size of the batch size used in each learning step
        seed (float): Random set for initialization
        """

        # Set class properties to input properties
        self.action_size = action_size
        self.batch_size  = batch_size
        self.seed        = random.seed(seed)

        # Create container for experiences
        self.memory = deque(maxlen = buffer_size)
        # Define format of experience type
        self.experience = namedtuple('Experience', ['state','action','reward','next_state','done'])


    def add(self,state, action, reward, next_state, done):
        """
        Creates an experience tuple with the current information and appends the element to replay memory

        Parameters
        ==========
        state (array_like): State before action
        action (array_like): Action that was taken
        reward (float): Reward for action
        next_state (array_like): State after action
        done (bool): Indicator if env was solved after action
        """

        self.memory.append(self.experience(state,action, reward, next_state,done))


    def get_sample(self):
        """
        Selects sample from exp buffer and returns values as PyTorch tensors
        """
        exp_selection = random.sample(self.memory,self.batch_size)

        # Init empty lists
        states      = []
        actions     = []
        rewards     = []
        next_states = []
        dones       = []

        # Append values from exp buffer
        for exp in exp_selection:
            if exp is not None:
                states.append(exp.state)
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                dones.append(exp.done)

        # Create PyTorch tensors from exp buffer
        states      = torch.from_numpy(np.vstack(states)).float().to(device)
        actions     = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards     = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones       = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def get_memory_size(self):
        """
        Returns current number of entries in memory
        """
        return len(self.memory)
