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

# Configuration
BATCH_SIZE = 64
EPOCHS_PER_TASK = 5
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/naive"

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    tasks = get_split_mnist_tasks(batch_size=BATCH_SIZE)
    num_tasks = len(tasks)
    
    # 2. Initialize Model (Binary classification for Split MNIST)
    # Note: Split MNIST usually re-maps labels to 0/1 for each task head
    # or uses a multi-head output. For simplicity in this naive baseline,
    # we use a shared head with 2 outputs (0 or 1) as defined in our dataloader.
    model = SmallCNN(num_classes=2).to(DEVICE)
    
    # Optimizer (re-initialized or kept across tasks? Naive usually keeps it)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Storage for Accuracy Matrix A[i, j]
    # Rows: Task trained (i), Cols: Task evaluated (j)
    acc_matrix = np.zeros((num_tasks, num_tasks))
    
    # 4. Sequential Training Loop
    for task_idx, task in enumerate(tasks):
        print(f"\n=== Training on Task {task_idx+1} (Digits {task['digits']}) ===")
        train_loader = task['train_loader']
        
        # Train
        for epoch in range(EPOCHS_PER_TASK):
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS_PER_TASK} - Loss: {avg_loss:.4f}")
            
        # Evaluate on ALL tasks (0 to 4) to fill row i
        print(f"--- Evaluating after Task {task_idx+1} ---")
        for eval_idx, eval_task in enumerate(tasks):
            acc = evaluate(model, eval_task['test_loader'], DEVICE)
            acc_matrix[task_idx, eval_idx] = acc
            print(f"Task {eval_idx+1} Acc: {acc:.2f}%")
            
    # 5. Save Results
    print("\nFinal Accuracy Matrix:")
    print(acc_matrix)
    np.save(f"{RESULTS_DIR}/acc_matrix.npy", acc_matrix)
    
    # Calculate Average Accuracy & Forgetting (at the end)
    # Avg Acc = Mean of the last row (performance on all tasks after training all)
    avg_acc = np.mean(acc_matrix[-1, :])
    
    # Forgetting for task j: max(A[:T-1, j]) - A[T-1, j]
    forgetting = []
    for j in range(num_tasks - 1): # Exclude last task
        max_prev = np.max(acc_matrix[:num_tasks-1, j])
        curr = acc_matrix[num_tasks-1, j]
        forgetting.append(max_prev - curr)
    
    avg_forgetting = np.mean(forgetting)
    
    results = {
        "avg_accuracy": avg_acc,
        "avg_forgetting": avg_forgetting,
        "matrix": acc_matrix.tolist()
    }
    
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAvg Accuracy (Final): {avg_acc:.2f}%")
    print(f"Avg Forgetting: {avg_forgetting:.2f}%")

if __name__ == "__main__":
    main()