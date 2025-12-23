import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm

from split_mnist import get_split_mnist_tasks
from cnn import SmallCNN
from eval import evaluate
from ewc import EWC # Import EWC

# Configuration
BATCH_SIZE = 64
EPOCHS_PER_TASK = 5
LR = 0.01
EWC_LAMBDA = 10000 # Try 1000-5000 range. Higher = more stability (less forgetting), lower = more plasticity.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/ewc"

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch_ewc(model, loader, optimizer, criterion, ewc):
    model.train()
    total_loss = 0
    total_ce = 0
    total_ewc = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Main classification loss
        ce_loss = criterion(outputs, targets)
        
        # EWC penalty
        ewc_loss = ewc.penalty_loss()
        
        loss = ce_loss + ewc_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_ewc += ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0

    return total_loss / len(loader)

def main():
    print(f"Using device: {DEVICE}")
    print(f"EWC Lambda: {EWC_LAMBDA}")
    
    tasks = get_split_mnist_tasks(batch_size=BATCH_SIZE)
    num_tasks = len(tasks)
    
    model = SmallCNN(num_classes=2).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize EWC
    ewc = EWC(model, importance_lambda=EWC_LAMBDA)
    
    acc_matrix = np.zeros((num_tasks, num_tasks))
    
    for task_idx, task in enumerate(tasks):
        print(f"\n=== Training on Task {task_idx+1} (Digits {task['digits']}) ===")
        train_loader = task['train_loader']
        
        # Train
        for epoch in range(EPOCHS_PER_TASK):
            avg_loss = train_one_epoch_ewc(model, train_loader, optimizer, criterion, ewc)
            # You might want to print just avg_loss or breakdown
            print(f"Epoch {epoch+1}/{EPOCHS_PER_TASK} - Loss: {avg_loss:.4f}")
            
        # REGISTER TASK FOR EWC
        # We calculate Fisher on the training data of the task just finished
        print("Computing Fisher Information...")
        ewc.register_task(train_loader, criterion)
        
        # Evaluate
        print(f"--- Evaluating after Task {task_idx+1} ---")
        for eval_idx, eval_task in enumerate(tasks):
            acc = evaluate(model, eval_task['test_loader'], DEVICE)
            acc_matrix[task_idx, eval_idx] = acc
            print(f"Task {eval_idx+1} Acc: {acc:.2f}%")
            
    # Save Results
    print("\nFinal Accuracy Matrix (EWC):")
    print(acc_matrix)
    np.save(f"{RESULTS_DIR}/acc_matrix.npy", acc_matrix)
    
    avg_acc = np.mean(acc_matrix[-1, :])
    forgetting = []
    for j in range(num_tasks - 1):
        max_prev = np.max(acc_matrix[:num_tasks-1, j])
        curr = acc_matrix[num_tasks-1, j]
        forgetting.append(max_prev - curr)
    
    avg_forgetting = np.mean(forgetting)
    
    results = {
        "avg_accuracy": avg_acc,
        "avg_forgetting": avg_forgetting,
        "matrix": acc_matrix.tolist(),
        "lambda": EWC_LAMBDA
    }
    
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAvg Accuracy (Final): {avg_acc:.2f}%")
    print(f"Avg Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()