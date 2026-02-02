# CIFAR-10 Image Classification with CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **CIFAR-10** dataset.

The goal is to build an end-to-end deep learning pipeline including:  
- Data loading and preprocessing  
- Model definition (CNN)  
- Training and evaluation  
- Visualization of sample images  

---

## Dataset

**CIFAR-10**  
- 50,000 training images  
- 10,000 test images  
- Image size: 32×32 (RGB)  
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

The dataset is automatically downloaded using `torchvision.datasets.CIFAR10`.

---

## Model Architecture

The model is a **Convolutional Neural Network (CNN)**:

- **Convolutional layers:**  
  - Conv1 → ReLU → MaxPool  
  - Conv2 → ReLU → MaxPool → Dropout  
- **Fully connected layers:**  
  - FC1 → ReLU → Dropout  
  - FC2 → Output layer (10 classes)

This structure allows the network to extract hierarchical image features and perform classification.

---

## Training Details

- **Framework:** PyTorch  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** SGD with momentum (0.9)  
- **Learning Rate:** 0.001  
- **Batch Size:** 64  
- **Epochs:** 10  

---

## Results

After training for 10 epochs:

| Metric         | Value      |
|----------------|------------|
| Final Loss     | ~1.08      |
| Test Accuracy  | **62.36%** |
| Train Accuracy | **64.99%** |

Training loss decreases steadily, showing proper convergence. Accuracy can be improved with deeper architectures or data augmentation.

---

## How to Run

1. Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```
