# MNIST with GAN (Digit Generation)

This project demonstrates how to generate synthetic handwritten digits using a Generative Adversarial Network (GAN) implemented in PyTorch.
The model is trained on the MNIST dataset and learns to produce realistic digit images through adversarial training between a generator and a discriminator.

--- 

## Dataset

- MNIST Handwritten Digits
- 60,000 training images
- Grayscale images of size 28×28
- 10 digit classes (0–9)
- Used in an unsupervised manner (labels are ignored)

---

## Preprocessing Steps

- Convert images to tensors
- Normalize pixel values to range [-1, 1]
- Flatten images for the discriminator

```
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

--- 

## Model Architecture
The GAN consists of two neural networks:

### Discriminator

| Layer     | Details            |
| --------- | ------------------ |
| Linear    | 784 → 1024         |
| LeakyReLU | slope = 0.2        |
| Linear    | 1024 → 512         |
| LeakyReLU | slope = 0.2        |
| Linear    | 512 → 256          |
| LeakyReLU | slope = 0.2        |
| Linear    | 256 → 1            |
| Sigmoid   | Output probability |

### Generator

| Layer  | Details           |
| ------ | ----------------- |
| Linear | 100 → 256         |
| ReLU   | Activation        |
| Linear | 256 → 512         |
| ReLU   | Activation        |
| Linear | 512 → 1024        |
| ReLU   | Activation        |
| Linear | 1024 → 784        |
| Tanh   | Output in [-1, 1] |

```
class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
```

--- 

## Training Setup

| Parameter            | Value                |
| -------------------- | -------------------- |
| Batch size           | 128                  |
| Noise vector (z_dim) | 100                  |
| Learning rate        | 0.0002               |
| Optimizer            | Adam                 |
| Loss function        | Binary Cross Entropy |
| Epochs               | 20                   |
| Adam betas           | (0.5, 0.999)         |

---

## Training Process

For each batch:
- Train Discriminator
- Real images labeled as 1
- Fake images labeled as 0
- Train Generator
- Try to fool discriminator
- Fake images labeled as real (1)
- This creates a minimax game:
- Discriminator tries to maximize accuracy
- Generator tries to minimize detection

### Sample Loss Progression

| Epoch | D Loss | G Loss |
| ----: | ------ | ------ |
|     1 | 0.83   | 1.84   |
|     5 | 0.40   | 3.77   |
|    10 | 0.78   | 2.82   |
|    15 | 0.72   | 1.36   |
|    20 | 0.78   | 1.70   |

The oscillating losses reflect typical GAN behavior.

--- 

## Testing & Evaluation

After training, random noise is passed into the generator:
```
z = torch.randn(16, z_dim)
sample_imgs = generator(z)
```
The generated digits are visualized using a grid.
The model successfully produces recognizable handwritten digits similar to MNIST.

---

## How to Run

- Install dependencies:
```
pip install torch torchvision matplotlib numpy
```

- Run the notebook:
```
jupyter notebook GAN.ipynb
```

---

## Key Concepts Demonstrated:
- Generative Adversarial Networks (GAN)
- Unsupervised learning
- Generator vs Discriminator dynamics
- Adversarial loss (BCE)
- Image normalization
- Noise vector sampling
- Image synthesis

--- 

## Limitations
- Fully connected architecture (no convolutions)
- No validation metrics (FID / IS)
- Training is unstable by nature
- Small number of epochs
- No model saving/loading
- Possible Improvements
- Use DCGAN (CNN-based) architecture
- Add label conditioning (cGAN)
- Train longer (50–100 epochs)
- Save generated samples per epoch
- Use Wasserstein GAN (WGAN)
- Track metrics like FID score

--- 

## Conclusion
This project is a minimal yet complete implementation of a GAN for image generation using PyTorch.
It demonstrates how adversarial learning can generate realistic data without explicit labels and provides a strong foundation for more advanced generative models such as:
DCGAN
Conditional GAN
StyleGAN
Diffusion models
It is an ideal entry-level project for understanding modern generative AI systems.
