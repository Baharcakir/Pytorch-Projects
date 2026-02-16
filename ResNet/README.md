# CIFAR-10 Classification with Custom ResNet (PyTorch)

This project implements a **custom Residual Neural Network (ResNet)** from scratch using PyTorch and trains it on the **CIFAR-10** dataset.

The goal is to understand how **residual blocks and skip connections** work internally, and how a deep CNN can be built without relying on pretrained models.

---

## Dataset

CIFAR-10 `(from torchvision.datasets)`

* 10 image classes
* 50,000 training images
* 10,000 test images
* RGB images of size **32×32**

Loaded via:

```python
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
```

---

## Preprocessing

Simple normalization is applied:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
```

**Why?**

* Converts images to tensors
* Normalizes pixel values to roughly `[-1, 1]`
* Helps stabilize training

No data augmentation is used in this experiment.

---

## Data Loaders

```python
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader  = DataLoader(testset,  batch_size=64, shuffle=False)
```

---

## Model Architecture

A **custom ResNet-like architecture** is implemented using:

* Residual blocks
* Skip connections
* Downsampling with stride
* Global average pooling

### Residual Block

Each block contains:

* Two 3×3 convolution layers
* Batch normalization
* ReLU activation
* Optional downsampling
* Identity skip connection

```python
out += identity  # skip connection
```

This allows gradients to flow easily in deep networks.

---

## Custom ResNet Structure

The network follows this structure:

```
Input (3×32×32)
↓
Conv7×7 + MaxPool
↓
[64]  Residual Blocks ×2
↓
[128] Residual Blocks ×2
↓
[256] Residual Blocks ×2
↓
[512] Residual Blocks ×2
↓
Adaptive Avg Pool
↓
Fully Connected (512 → 10)
```

This is similar in spirit to **ResNet-18**, but adapted for CIFAR-10.

---

## Alternative: Pretrained ResNet18

The code also supports transfer learning:

```python
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(num_ftrs,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
```

This allows switching between:

* Custom architecture
* Pretrained ImageNet backbone

---

## Training Setup

| Parameter     | Value            |
| ------------- | ---------------- |
| Dataset       | CIFAR-10         |
| Classes       | 10               |
| Batch size    | 64               |
| Optimizer     | Adam             |
| Learning rate | 0.001            |
| Loss          | CrossEntropyLoss |
| Epochs        | 1                |
| Device        | CPU / CUDA       |

---

## Training Loop

At each iteration:

* Forward pass
* Compute cross-entropy loss
* Backpropagation
* Update weights

```python
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

Progress is visualized using `tqdm`.

---

## Evaluation

After training:

```python
model.eval()
```

The model is tested on unseen data:

```python
Test accuracy: 61.65%
```

---

## Results

| Metric        | Value  |
| ------------- | ------ |
| Training loss | 1.36   |
| Test accuracy | 61.65% |
| Epochs        | 1      |

This is **reasonable performance for a custom CNN trained from scratch in one epoch**.

---

## Key Concepts Demonstrated

* Residual Networks (ResNet)
* Skip connections
* Custom CNN architecture
* Downsampling with stride
* Batch normalization
* GPU training with CUDA
* Image classification
* PyTorch model design

---

## Limitations

* Only 1 epoch of training
* No data augmentation
* No learning rate scheduler
* No confusion matrix
* No validation split
* No hyperparameter tuning

---

## Possible Improvements

* Train for 10–50 epochs
* Add data augmentation
* Use LR scheduling
* Add dropout
* Track validation accuracy
* Compare with pretrained ResNet18
* Compute Top-5 accuracy
* Add confusion matrix

---

## How to Run

Install dependencies:

```bash
pip install torch torchvision tqdm
```

Run:

```bash
python resnet_cifar10.py
```

---

## Conclusion

This project shows how a **ResNet can be built from scratch** using only basic PyTorch layers.

It demonstrates that:

* Skip connections make deep models trainable
* Even simple implementations can reach >60% accuracy
* Understanding architecture internals is crucial before using large pretrained models

This is an excellent foundation for:

* Learning modern CNN design
* Implementing ResNet variants
* Transitioning to advanced architectures like:

  * ResNet50
  * DenseNet
  * EfficientNet
  * Vision Transformers
