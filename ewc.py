import torch
import torch.nn as nn
import copy

class EWC:
    def __init__(self, model, importance_lambda=1000):
        self.model = model
        self.importance_lambda = importance_lambda  # Controls strength of regularization
        self.tasks_params = [] # List of (fisher_dict, opt_param_dict) for each previous task
        self.device = next(model.parameters()).device

    def register_task(self, dataloader, criterion):
        """
        Called AFTER training a task.
        Computes Fisher Information and stores optimal parameters.
        """
        self.model.eval()
        fisher = {}
        opt_params = {}
        
        # Initialize Fisher dict with zeros matching model params
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
                opt_params[name] = param.data.clone() # Store θ*
        
        # Compute Fisher Information (gradients^2)
        # We iterate over the dataset to estimate expectation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Using negative log likelihood (CrossEntropy)
            # We can sample labels from the model distribution (theoretically correct for Fisher)
            # OR use ground truth labels (empirical Fisher, common approximation).
            # The assignment mentions: E[ (d/dtheta log p(y|x))^2 ]
            
            # Empirical Fisher (using ground truth y):
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Accumulate sum of squared gradients
                    fisher[name] += param.grad.data ** 2
        
        # Normalize by number of batches (or samples, depending on implementation preference)
        # Here we just average over batches
        num_batches = len(dataloader)
        for name in fisher:
            fisher[name] /= num_batches
            
        self.tasks_params.append((fisher, opt_params))
        print(f"EWC: Task registered. Total tasks stored: {len(self.tasks_params)}")

    def penalty_loss(self):
        """
        Computes the EWC regularization loss:
        L_ewc = (lambda/2) * sum_i ( F_i * (theta_i - theta_i*)^2 )
        Summed over all previous tasks.
        """
        loss = 0
        for fisher, opt_params in self.tasks_params:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # (theta - theta*)
                    delta = param - opt_params[name]
                    # Fisher * delta^2
                    loss += (fisher[name] * (delta ** 2)).sum()
        
        return (self.importance_lambda / 2) * loss