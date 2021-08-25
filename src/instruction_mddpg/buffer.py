""" Packaged MDDPG"""
from typing import Dict, List, Deque, Tuple
from collections import deque
import numpy as np
from collections import deque, namedtuple

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, action_size, instruction_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.
        
        :buffer_size (int): maximum size of buffer
        :batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.instruction_size = instruction_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "instruction", "action", "reward", "next_state", "next_instruction", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, instruction, action, reward, next_state, next_instruction, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, instruction, action, reward, next_state, next_instruction, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        instructions = torch.from_numpy(np.vstack([e.instruction for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_instructions = torch.from_numpy(np.vstack([e.next_instruction for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, instructions, actions, rewards, next_states, next_instructions, dones)
    
    def is_ready(self):
        return len(self.memory) > self.batch_size

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)