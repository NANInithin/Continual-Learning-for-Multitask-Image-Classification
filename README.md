# Continual Learning for Multitask Image Classification

A comprehensive implementation and evaluation of continual learning (CL) methods for sequential multitask classification on Split MNIST. This project compares three approaches: **Sequential Fine-Tuning (Naive CL)**, **Elastic Weight Consolidation (EWC)**, and **Experience Replay**, demonstrating how replay-based methods effectively mitigate catastrophic forgetting.

## 🎯 Project Overview

**Continual Learning Challenge**: How can neural networks learn multiple tasks sequentially without "forgetting" previously learned knowledge?

This project addresses catastrophic forgetting through:
- ✅ **Naive Baseline**: Sequential fine-tuning (lower bound)
- ✅ **Regularization Method**: EWC with Fisher Information Matrix
- ✅ **Replay Method**: Experience Replay buffer with empirical validation
- ✅ **Rigorous Evaluation**: Accuracy matrix, forgetting metrics, comparative analysis

**Key Results**:
| Method | Final Accuracy | Avg Forgetting |
|--------|---|---|
| Naive CL | 66.67% | 41.50% |
| EWC (λ=2000) | 70.12% | 37.14% |
| **Experience Replay** | **97.80%** | **1.84%** |

---

## 📦 Repository Structure

```
Continual-Learning-for-Multitask-Image-Classification/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
│
├── cnn.py                             # Small CNN architecture
├── eval.py                            # Evaluation metrics
├── ewc.py                             # Elastic Weight Consolidation
├── replay.py                          # Experience Replay buffer
├── split_mnist.py                     # Split MNIST dataset
│
├── train_naive.py                     # Naive sequential fine-tuning
├── train_ewc.py                       # EWC training script
├── train_replay.py                    # Experience Replay training
├── plot_results.py                    # Generate comparison plots
│
├── results/                           # Experiment outputs
│   ├── naive/
│   │   ├── acc_matrix.npy
│   │   └── metrics.json
│   ├── ewc/
│   │   ├── acc_matrix.npy
│   │   └── metrics.json
│   └── replay/
│       ├── acc_matrix.npy
│       └── metrics.json
│
├── docs/
│   └── REPORT.md                      # Full 2-4 page research report
│
└── data/                              # MNIST dataset (auto-downloaded)
    └── MNIST/
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA 12.4+** (optional, for GPU acceleration)
- **GPU**: RTX 4060 or equivalent (recommended, CPU also works)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/NANInithin/Continual-Learning-for-Multitask-Image-Classification.git
cd Continual-Learning-for-Multitask-Image-Classification
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -U pip
pip install -r requirements.txt

# For GPU (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

4. **Verify GPU access (optional)**:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 📊 Running Experiments

### 1. Naive Sequential Fine-Tuning (Baseline)
```bash
python train_naive.py
```
**Output**: Results saved to `results/naive/`

### 2. Elastic Weight Consolidation (EWC)
```bash
python train_ewc.py
```
**Hyperparameter**: Adjust `EWC_LAMBDA` in `train_ewc.py` (default: 2000)

### 3. Experience Replay
```bash
python train_replay.py
```
**Configuration**: Modify `SAMPLES_PER_TASK` in `train_replay.py` (default: 200)

### 4. Generate Comparison Plots
```bash
python plot_results.py
```
**Outputs**:
- `comparison_accuracy.png` – Accuracy curves across tasks
- `comparison_forgetting.png` – Average forgetting bar chart

---

## 📈 Results Visualization

The project generates two key comparison plots:

### Figure 1: Average Accuracy vs Tasks
Shows how each method maintains accuracy as new tasks are learned. Experience Replay maintains ~99% accuracy across all tasks, while naive fine-tuning drops significantly.

### Figure 2: Average Forgetting Comparison
Bar chart comparing forgetting across methods. Replay achieves 20× lower forgetting than naive CL (1.84% vs 41.50%).

---

## 🔬 Methodology

### Dataset: Split MNIST
- **5 sequential binary tasks** (digits {0,1}, {2,3}, {4,5}, {6,7}, {8,9})
- ~12,600 training samples per task
- Labels remapped to binary (0 or 1) within each task

### Model: Small CNN
```
Input (1×28×28)
  → Conv2d(1, 32, 3×3) + MaxPool2d(2×2)
  → Conv2d(32, 64, 3×3) + MaxPool2d(2×2)
  → Dropout(0.25)
  → Linear(64×7×7, 128) + ReLU
  → Linear(128, 2)
Output (binary classification)
```
**Total Parameters**: ~230K

### Training Protocol
- **Optimizer**: SGD (momentum=0.9)
- **Learning Rate**: 0.01
- **Epochs per Task**: 5
- **Batch Size**: 64
- **Evaluation**: After each task, evaluate on all previously seen + current tasks

### Continual Learning Methods

#### 1. Sequential Fine-Tuning (Naive CL)
No explicit mechanism to prevent forgetting. Serves as lower bound baseline.

#### 2. Elastic Weight Consolidation (EWC)
Regularizes parameter updates based on Fisher Information Matrix:
```
L_EWC = L_task(θ) + (λ/2) × Σ_i [ (Σ_k F_i^(k)) × (θ_i - θ_i*)^2 ]
```
- **λ = 2000**: Stability-plasticity trade-off parameter
- **Fisher**: Diagonal approximation computed empirically on training data
- **Memory**: No data storage required

#### 3. Experience Replay
Maintains buffer of past task samples, replayed during new task training:
```
L_Replay = E_{(x,y) ~ D_t ∪ M} [ CrossEntropy(f_θ(x), y) ]
```
- **Buffer Size**: 200 samples per task (1,000 total)
- **Update**: Random sampling from each past task after completion

### Evaluation Metrics

**Accuracy Matrix** `A_{i,j}`: Accuracy on task j after training on task i

**Average Accuracy** (final): `AvgAcc = (1/T) × Σ_j A_{T,j}`

**Forgetting per Task**: `F_j = max_{i<T} A_{i,j} - A_{T,j}`

**Overall Forgetting**: `F = (1/(T-1)) × Σ_j F_j`

---

## 📚 Key Findings

1. **Experience Replay Dominates**: 97.80% accuracy with 1.84% forgetting
2. **EWC Limited by Shared Head**: Label remapping creates output conflicts EWC cannot resolve
3. **Catastrophic Forgetting Severe**: Naive approach drops to 5.91% on Task 1 by end
4. **Replay Overhead Negligible**: 1,000 samples (~13 MB) for ~97% accuracy improvement

---

## 📖 Documentation

- **Full Report**: See `docs/REPORT.md` for comprehensive 2-4 page analysis
- **Code Comments**: All modules well-documented with docstrings
- **Evaluation**: See results/ directory for detailed metrics

---

## 🧪 Code Quality

- ✅ Modular, well-documented architecture
- ✅ PyTorch best practices
- ✅ Reproducible experiments
- ✅ Clear separation of concerns
- ✅ Comprehensive error handling

---

## 🔧 Configuration

Main hyperparameters in individual scripts:

```python
# Data
BATCH_SIZE = 64
NUM_WORKERS = 2

# Training
EPOCHS_PER_TASK = 5
LR = 0.01
MOMENTUM = 0.9

# EWC
EWC_LAMBDA = 2000

# Replay
SAMPLES_PER_TASK = 200

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@project{continual_learning_2025,
  author = {NANInithin},
  title = {Continual Learning for Multitask Image Classification},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/NANInithin/Continual-Learning-for-Multitask-Image-Classification}}
}
```

---

## 📚 References

- **Assignment**: TP2 - Advances in Machine Vision, Paris Saclay University
- **EWC**: Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS, 114(13), 3521–3526.
- **MNIST**: LeCun et al. (1998). *Gradient-based learning applied to document recognition*. IEEE, 86(11), 2278–2324.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Task-specific output heads (multi-head architecture)
- Additional CL methods (Progressive Neural Networks, PackNet, etc.)
- Alternative benchmarks (Permuted MNIST, CIFAR-100)
- Hyperparameter optimization framework
- Extended analysis notebook

---

## 📄 License

MIT License – See LICENSE file for details

---

## 👤 Author

**NANInithin**  
Paris Saclay University, Master's in Machine Vision and AI  
December 2025

---

## ⚡ Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` to 32 or 16 |
| Slow training | Ensure GPU is being used: `torch.cuda.is_available()` |
| Results don't match report | Check random seed, hardware differences may cause ~1% variance |
| Missing MNIST data | Scripts auto-download on first run; ensure internet connection |

---

## 📞 Support

For issues, questions, or suggestions:
1. Check existing GitHub Issues
2. Review documentation in `docs/REPORT.md`
3. Check code comments and docstrings
4. Open a new Issue with detailed description

---

**Last Updated**: December 2025  
**Status**: ✅ Complete & Ready for Submission
