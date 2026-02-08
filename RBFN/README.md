# Iris Classification with Radial Basis Function Network (RBFN)

This project demonstrates how to implement a **Radial Basis Function Network (RBFN)** from scratch using PyTorch and apply it to a classical supervised learning problem: Iris flower classification.

The model learns nonlinear decision boundaries using Gaussian radial basis functions and achieves high accuracy on a small real-world dataset.

---

## Dataset

Iris Dataset (UCI Machine Learning Repository)
150 samples
4 numerical features: 
- Sepal length
- Sepal width
- Petal length
- Petal width

3 classes:
- Setosa
- Versicolor
- Virginica

Loaded via ucimlrepo:
```
iris = fetch_ucirepo(id=53)
X = iris.data.features.values
y = iris.data.targets.values.ravel()
```

---

## Preprocessing Steps

- Encode string labels to integers
- Standardize features using Z-score normalization
- Split into train and test sets (70% / 30%)
```
y, _ = pd.factorize(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

---

## Model Architecture

The RBF Network consists of three main components:
- Input Layer
- RBF Hidden Layer (Gaussian kernels)
- Linear Output Layer

RBF Kernel $\phi(x, c) = e^{-\beta \|x - c\|^2}$

Implemented as:
```
def rbf_kernel(X, centers, beta):
    return torch.exp(-beta * torch.cdist(X, centers)**2)
RBFN Class
class RBFN(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.beta = nn.Parameter(torch.ones(1) * 2.0)
        self.linear = nn.Linear(num_centers, output_dim)

    def forward(self, X):
        phi = rbf_kernel(X, self.centers, self.beta)
        return self.linear(phi)
```

---

## Training Setup
| Parameter | Value |
| :---        |    :----:   |
| Input features | 4 |
| RBF centers	| 10 |
| Output classes	| 3 | 
| Optimizer	| Adam |
| Learning rate |	0.01 |
| Loss function |	Cross Entropy |
| Epochs | 100 |

---

## Training Process
At each epoch:
- Compute RBF activations
- Pass through linear layer
- Compute cross-entropy loss
- Backpropagate gradients
- Update centers, beta, and weights

### Sample Loss Progression

| Epoch |	Loss |
| :---        |    :----:   |
| 10 |	1.09 |
| 30 |	1.05 |
| 50 | 0.86 | 
| 70 |	0.59 |
| 100	| 0.30 |

Loss decreases smoothly, indicating effective learning.

---

## Testing & Evaluation

```
with torch.no_grad():
    y_pred = model(X_test)
    accuracy = (torch.argmax(y_pred, axis=1) == y_test).float().mean()
``` 

Final Accuracy
Accuracy: 95.5%

This is comparable to classical ML models (SVM, KNN) on the Iris dataset.

---

## How to Run

- Install dependencies:
```
pip install torch torchvision numpy pandas scikit-learn ucimlrepo
```

- Run the notebook:
```
jupyter notebook RBFN.ipynb
```
---

## Key Concepts Demonstrated

- Radial Basis Function Networks
- Gaussian kernels
- Nonlinear feature mapping
- Learnable kernel centers
- PyTorch custom modules
- Multiclass classification
- Z-score normalization
- Train/test evaluation

---

## Limitations

- Centers initialized randomly (no K-means)
- Single hidden RBF layer
- No regularization
- No GPU usage
- No model persistence
- Possible Improvements
- Initialize centers using K-means
- Use separate beta per center
- Add validation split
- Plot decision boundaries (2D PCA)
- Save/load trained model
- Compare with:
  - MLP
  - SVM
  - KNN

---

## Conclusion

This project provides a minimal yet complete implementation of an RBF Network in PyTorch and shows that kernel-based neural networks can achieve strong performance even on small datasets.

It is an excellent bridge between:
- Classical kernel methods (RBF, SVM)
- Modern deep learning frameworks
And a great conceptual stepping stone toward:
- Kernelized neural networks
- Gaussian Processes
- Hybrid deep–kernel models

Perfect for understanding nonlinear representation learning beyond MLPs.
