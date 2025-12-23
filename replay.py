import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, samples_per_task=200, device='cpu'):
        """
        Stores 'samples_per_task' images and labels from each task.
        """
        self.samples_per_task = samples_per_task
        self.device = device
        # List to store batches of memory: [(images_t1, labels_t1), (images_t2, labels_t2), ...]
        self.memory = [] 

    def add_data(self, dataloader):
        """
        Called after finishing a task. Selects random samples to store.
        """
        # 1. Collect all data from the loader
        all_x, all_y = [], []
        for x, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)
        
        # 2. Randomly select indices
        num_samples = all_x.size(0)
        # Ensure we don't try to take more than exists
        n_select = min(num_samples, self.samples_per_task)
        indices = torch.randperm(num_samples)[:n_select]
        
        # 3. Store in memory
        self.memory.append((all_x[indices].to(self.device), all_y[indices].to(self.device)))
        print(f"Replay: Added {n_select} samples to buffer. Total tasks in memory: {len(self.memory)}")

    def sample(self, batch_size):
        """
        Returns a batch of data sampled from the memory.
        """
        if not self.memory:
            return None, None
            
        # Strategy: Flatten all memory and sample uniformly
        # (For 5 tasks x 200 samples = 1000 items, this is cheap)
        all_x_mem = torch.cat([m[0] for m in self.memory])
        all_y_mem = torch.cat([m[1] for m in self.memory])
        
        # Random indices for the batch
        indices = torch.randint(0, len(all_x_mem), (batch_size,))
        
        return all_x_mem[indices], all_y_mem[indices]
