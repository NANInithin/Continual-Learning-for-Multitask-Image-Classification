import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_data(method_name, path):
    try:
        data = np.load(f"{path}/acc_matrix.npy")
        with open(f"{path}/metrics.json", "r") as f:
            metrics = json.load(f)
        return data, metrics
    except FileNotFoundError:
        print(f"Warning: No results found for {method_name}")
        return None, None

def plot_accuracy_curve(methods_data):
    plt.figure(figsize=(10, 6))
    
    for name, (matrix, _) in methods_data.items():
        if matrix is not None:
            # Average accuracy after each task (mean of the relevant row section? 
            # Usually: Mean accuracy on "seen tasks" OR Mean accuracy on "all tasks")
            # The PDF formula for AvgAcc is 1/T * sum(A_T,j) -> Final Avg Acc.
            # For the curve, let's plot "Average Accuracy on All Tasks" after training task i.
            
            # Row i contains results on all tasks 0..4 after training task i
            # avg_acc_history = [np.mean(matrix[i, :]) for i in range(5)] # Mean over all tasks
            
            # Alternatively: Mean over tasks seen SO FAR (common in CL papers)
            # avg_acc_history = [np.mean(matrix[i, :i+1]) for i in range(5)]
            
            # Let's stick to Mean over ALL tasks (to show 0 performance on future tasks rising, or just global performance)
            avg_acc_history = np.mean(matrix, axis=1) 
            
            plt.plot(range(1, 6), avg_acc_history, marker='o', label=name)

    plt.title('Average Accuracy vs Tasks')
    plt.xlabel('Task Trained')
    plt.ylabel('Average Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_accuracy.png')
    print("Saved comparison_accuracy.png")

def plot_forgetting_bar(methods_data):
    names = []
    forgetting_vals = []
    
    for name, (_, metrics) in methods_data.items():
        if metrics:
            names.append(name)
            forgetting_vals.append(metrics['avg_forgetting'])
            
    plt.figure(figsize=(8, 6))
    plt.bar(names, forgetting_vals, color=['red', 'blue', 'green'])
    plt.title('Average Forgetting (Lower is Better)')
    plt.ylabel('Forgetting (%)')
    plt.grid(axis='y')
    plt.savefig('comparison_forgetting.png')
    print("Saved comparison_forgetting.png")

def main():
    methods = {
        "Naive": load_data("Naive", "results/naive"),
        "EWC":   load_data("EWC", "results/ewc"),
        "Replay": load_data("Replay", "results/replay")
    }
    
    # Filter out missing runs
    methods = {k: v for k, v in methods.items() if v[0] is not None}
    
    plot_accuracy_curve(methods)
    plot_forgetting_bar(methods)

if __name__ == "__main__":
    main()
