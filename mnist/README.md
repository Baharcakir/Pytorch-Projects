# MNIST Number Classification with ANN (PyTorch)

This project implements a simple **Artificial Neural Network (ANN)** using **PyTorch** to classify handwritten digits from the MNIST dataset.

The goal is to build an end-to-end deep learning pipeline including:
- Data loading and preprocessing  
- Model definition  
- Training  
- Evaluation  

---

## Dataset

**MNIST**  
- 60,000 training images  
- 10,000 test images  
- Image size: 28×28 (grayscale)  
- 10 classes (digits 0–9)

The dataset is automatically downloaded using `torchvision.datasets.MNIST`.

---

## Model Architecture

The model is a fully connected feed-forward neural network:


---

## Training Details

- **Framework:** PyTorch  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning Rate:** 0.001  
- **Batch Size:** 64  
- **Epochs:** 10  

---

## Results

After training for 10 epochs:

| Metric        | Value |
|--------------|-------|
| Final Loss   | ~0.065 |
| Test Accuracy| **97.13%** |

Training loss decreases steadily, showing stable convergence.

---

## Sample Output

The notebook also visualizes sample digits from the dataset:

- Displays random MNIST images
- Shows their true labels
- Plots training loss per epoch

---

## How to Run

1. Install dependencies:
```bash
pip install torch torchvision matplotlib
```

## What This Project Demonstrates
This project shows:
- Basic PyTorch workflow
- GPU/CPU device handling
- Training and evaluation loops
- Clean modular code structure
- It is a solid baseline for more advanced models such as CNNs.

## Possible Improvements
Future extensions could include:
- Adding convolutional layers (CNN)
- Using dropout and batch normalization
- Hyperparameter tuning
- Confusion matrix visualization
