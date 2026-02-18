# Sentiment Classification with Transformers (PyTorch)

This project demonstrates how to build a simple **binary sentiment classifier** from scratch using a **Transformer-based neural network** in PyTorch.

The goal is to classify short English sentences as either **positive (1)** or **negative (0)** using a small custom dataset and a lightweight Transformer model.

---

## Dataset

A manually created dataset consisting of:

* **100 positive sentences**
* **100 negative sentences**
* Total: **200 samples**

Examples:

**Positive**

* `"this is amazing"`
* `"i really love this product"`
* `"highly recommended"`

**Negative**

* `"this is terrible"`
* `"waste of time"`
* `"i am very disappointed"`

Labels:

* `1` → Positive
* `0` → Negative

---

## Preprocessing

Each sentence goes through:

1. Lowercasing
2. Punctuation removal
3. Tokenization (split by spaces)

```python
def preprocess(text):
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text
```

---

## Vocabulary & Encoding

A simple **word-level vocabulary** is built using `Counter`.

* Index starts from 1
* `0` is reserved for **padding**

```python
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.items())}
vocab[""] = 0
```

Each sentence is converted to a fixed-length tensor:

* `max_len = 15`
* Short sentences → padded
* Long sentences → truncated

```python
def sentence_to_tensor(sentence, vocab, max_len=15):
  tokens = sentence.split()
  indices = [vocab.get(word, 0) for word in tokens]
  indices = indices[:max_len]
  indices += [0] * (max_len - len(indices))
  return torch.tensor(indices)
```

---

## Train / Test Split

Data is split using scikit-learn:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* 80% training
* 20% testing

---

## Model Architecture

A custom **Transformer-based classifier**:

### Components

* Word Embedding
* Learnable Positional Encoding
* PyTorch `nn.Transformer`
* Fully connected classifier

### Architecture Summary

| Layer               | Description               |
| ------------------- | ------------------------- |
| Embedding           | `(vocab_size → 32)`       |
| Positional Encoding | `(1 × 15 × 32)`           |
| Transformer         | 4 encoder layers, 4 heads |
| FC                  | `(32×15 → 64)`            |
| Output              | `(64 → 1)`                |
| Activation          | Sigmoid                   |

### Model Definition

```python
class Transformer(nn.Module):
  def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
    self.transformer = nn.Transformer(
        d_model=embedding_dim,
        nhead=num_heads,
        num_encoder_layers=num_layers,
        dim_feedforward=hidden_dim
    )
    self.fc = nn.Linear(embedding_dim * max_len, hidden_dim)
    self.out = nn.Linear(hidden_dim, num_classes)
    self.sigmoid = nn.Sigmoid()
```

---

## Training Setup

| Parameter      | Value                 |
| -------------- | --------------------- |
| Model          | Custom Transformer    |
| Embedding Dim  | 32                    |
| Heads          | 4                     |
| Encoder Layers | 4                     |
| Hidden Dim     | 64                    |
| Optimizer      | Adam                  |
| Learning Rate  | 0.0005                |
| Loss           | BCELoss               |
| Epochs         | 100                   |
| Task           | Binary Classification |

---

## Training Loop

At each epoch:

* Forward pass
* Binary cross-entropy loss
* Backpropagation
* Weight update

```python
output = model(X_train.long()).squeeze()
loss = criterion(output, y_train.float())
loss.backward()
optimizer.step()
```

Loss gradually decreases from ~0.69 to ~0.32.

---

## Evaluation

Predictions are thresholded:

```python
y_pred = (y_pred > 0.5).float()
```

Metrics:

```python
accuracy_score(y_test, y_pred)
accuracy_score(y_train, y_pred_training)
```

---

## Results

```
Test Accuracy: 67.5%
Train Accuracy: 90.2%
```

### Interpretation

* The model **overfits**:

  * Very high training accuracy
  * Much lower test accuracy

This is expected due to:

* Very small dataset
* No regularization
* No dropout
* No validation set

---

## Key Concepts Demonstrated

* Text preprocessing
* Vocabulary building
* Word embeddings
* Positional encoding
* Transformer encoder
* Binary classification
* Overfitting in deep learning
* End-to-end NLP pipeline in PyTorch

---

## Limitations

* Extremely small dataset (200 samples)
* No validation set
* No dropout or regularization
* No batching (full dataset each step)
* Word-level tokens only
* No attention masking
* No GPU usage
* No early stopping

---

## Possible Improvements

* Add **dropout layers**
* Use **mini-batches**
* Add **validation set**
* Use **LayerNorm**
* Use **BERT or DistilBERT**
* Use **pretrained embeddings (GloVe, FastText)**
* Increase dataset size
* Add **confusion matrix**
* Use **F1-score instead of only accuracy**

---

## How to Run

Install dependencies:

```bash
pip install torch torchvision scikit-learn
```

Run the script or notebook:

```bash
Transformers.ipynb
```

---

## Conclusion

This project shows a **minimal Transformer for NLP** built completely from scratch.

Despite the simplicity:

* The model successfully learns sentiment patterns
* Demonstrates core Transformer mechanics
* Provides a strong educational baseline

This type of architecture is the foundation of:

* BERT
* GPT
* RoBERTa
* T5

And modern NLP systems used in:

* Chatbots
* Recommendation systems
* Opinion mining
* Review analysis
* Social media sentiment tracking

A perfect stepping stone toward real-world NLP models 🚀
