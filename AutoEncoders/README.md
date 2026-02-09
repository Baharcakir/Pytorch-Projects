# FashionMNIST Autoencoder with PyTorch

This project demonstrates how to implement a **fully connected Autoencoder** using PyTorch and apply it to an unsupervised learning task: **image reconstruction** on the FashionMNIST dataset.

The model learns a compressed latent representation of images and reconstructs them while minimizing reconstruction error.

---

## Dataset

FashionMNIST (from `torchvision.datasets`)

- 60,000 training images  
- 10,000 test images  
- Image size: 28 × 28 grayscale  
- 10 classes (used only for loading, not for training)

Each image represents a clothing item such as:
- T-shirt/top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle boot  

Loaded via:
```
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
```
---

## Preprocessing Steps

- Convert images to PyTorch tensors
- Normalize pixel values to range [0, 1] using ToTensor()
- No labels are used (unsupervised learning)
` transform = transforms.Compose([transforms.ToTensor()]) `

---

## Model Architecture

The Autoencoder consists of two main parts:
- Encoder
  - Compresses the input image into a low-dimensional latent vector.
- Decoder
  - Reconstructs the original image from the latent representation.

---

## Architecture:

| Layer     | Input → Output |
| --------- | -------------- |
| Flatten   | 1×28×28 → 784  |
| Linear    | 784 → 256      |
| ReLU      | -              |
| Linear    | 256 → 64       |
| ReLU      | -              |
| Linear    | 64 → 256       |
| ReLU      | -              |
| Linear    | 256 → 784      |
| Sigmoid   | -              |
| Unflatten | 784 → 1×28×28  |

Implemented as:

```
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

---

## Training Setup

| Parameter        | Value                     |
| ---------------- | ------------------------- |
| Batch size       | 128                       |
| Latent dimension | 64                        |
| Optimizer        | Adam                      |
| Learning rate    | 0.001                     |
| Loss function    | Mean Squared Error (MSE)  |
| Epochs           | 5                         |
| Device           | CPU / CUDA (if available) |

---

## Training Process

At each epoch:
- Forward pass through encoder and decoder
- Compute reconstruction loss (MSE)
- Backpropagate gradients
- Update model weights

`
loss = criterion(outputs, inputs)
loss.backward()
optimizer.step()
`


## Early Stopping

A simple Early Stopping mechanism is implemented to prevent overfitting.

Stops training if loss does not improve by at least **min_delta** for **patience** epochs.


`early_stopping = EarlyStopping(patience=5, min_delta=0.001)`

---

## Evaluation Method

Reconstruction quality is evaluated using:

**Structural Similarity Index (SSIM)**

SSIM measures perceptual similarity between two images:
- 1 → identical images
- 0 → no similarity

Custom SSIM implementation using Gaussian filtering.

`ssim_score = compute_ssim(original, reconstructed)`
`

---

## Visualization

The evaluation function:
- Displays original images (top row)
- Displays reconstructed images (bottom row)
- Computes average SSIM score

`evaluate(model, test_loader, n_images=10)`

### Sample Output

```
epoch 1/5, loss 0.0452  
epoch 2/5, loss 0.0321  
epoch 3/5, loss 0.0287  
epoch 4/5, loss 0.0263  
epoch 5/5, loss 0.0251  

Average SSIM: 0.81
```

This indicates good reconstruction quality for a simple fully connected autoencoder.

---

## How to Run

- Install dependencies:
```
pip install torch torchvision numpy pandas matplotlib scipy
```
- Run the notebook:
```
 autoencoder.py
```

---

## Key Concepts Demonstrated

- Autoencoders
- Unsupervised learning
- Dimensionality reduction
- Image reconstruction
- Latent space representation
- Mean Squared Error loss
- Early stopping
- Structural Similarity Index (SSIM)
- PyTorch custom models

---

## Limitations

- Fully connected architecture (no CNN)
- No validation split
- No model saving/loading
- No denoising or regularization
- Latent space not visualized

---

## Possible Improvements

- Use Convolutional Autoencoder (CAE)
- Add Denoising Autoencoder
- Visualize latent space using PCA or t-SNE
- Add validation set
- Save reconstructed images to disk
- Train for more epochs
- Use Variational Autoencoder (VAE)

---

## Conclusion

This project provides a clean and minimal implementation of an Autoencoder in PyTorch and shows how unsupervised neural networks can learn meaningful compressed representations of image data.
It is an excellent foundation for:
- Representation learning
- Anomaly detection
- Dimensionality reduction
- Generative models

And a strong stepping stone toward:
- Convolutional Autoencoders
- Variational Autoencoders
- Deep generative modeling
