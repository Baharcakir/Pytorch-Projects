# Custom Dataset with LSTM (Next Word Prediction)

This project demonstrates how to build a simple **word-level language model** using an **LSTM (Long Short-Term Memory)** network in **PyTorch**.  
The model is trained on a small custom Turkish text dataset and learns to predict the **next word** in a sequence.

The project also includes a simple **grid search–based hyperparameter tuning** process and a final evaluation through text generation.

---

## Dataset

**Custom Text Dataset**

The dataset consists of a single Turkish product review:

> *"Bu ürün beklentimi fazlasıyla karşıladı. Malzeme kalitesi gerçekten çok iyi. Kargo hızlı ve sorunsuz bir şekilde elime ulaştı. Fiyatına göre performansı harika. Kesinlikle tavsiye ederim!"*

### Preprocessing Steps

1. Lowercase conversion  
2. Remove punctuation  
3. Tokenize into words  
4. Build vocabulary  
5. Map words to indices  

Example:

```python
words = text.replace(".", "").replace("!","").lower().split()
```

---

Training pairs are created as:
(bu → ürün)
(ürün → beklentimi)
(beklentimi → fazlasıyla)

This forms a sequence-to-sequence learning problem.

---

## Model Architecture
The model is a simple LSTM-based language model:
| Layer     | Details                          |
|-----------|----------------------------------|
| Embedding | vocab_size → embedding_dim       |
| LSTM      | embedding_dim → hidden_dim       |
| Linear    | hidden_dim → vocab_size          |

```
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
```

---

## Hyperparameter Tuning
A small grid search is applied:
- Embedding sizes: 8, 16
- Hidden sizes: 32, 64
- Learning rates: 0.01, 0.005

Total combinations: 8 experiments

Each configuration is trained for 50 epochs and evaluated using **CrossEntropyLoss**.

---

## Best Parameters Found
- Embedding size: 8  
- Hidden size: 32  
- Learning rate: 0.01

---

# Final Training
Using the best hyperparameters, the final model is trained for 100 epochs.

### Sample loss progression:
| Epoch | Loss  |
|-------|-------|
| 0     | 79.29 |
| 20    | 0.48  |
| 50    | 0.08  |
| 90    | 0.027 |

This indicates the model successfully memorizes the word transitions.

---

## Testing & Evaluation
A simple text generation function is used:
pred_seq(start_word="ürün", num_words=10)

### Output Example
ürün beklentimi fazlasıyla karşıladı malzeme kalitesi gerçekten çok iyi kargo hızlı
The model reproduces a grammatically coherent and semantically correct continuation.

---

## How to Run
Install dependencies:
```
pip install torch torchvision matplotlib numpy
```
Run the script or notebook:
```
jupyter notebook LSTM.ipynb
```

---

## Key Concepts Demonstrated
- Word-level tokenization
- Vocabulary building
- Embedding layers
- LSTM sequence modeling
- Cross-entropy loss
- Hyperparameter grid search
- Next-word prediction

---

## Limitations
- Dataset is extremely small
- Model memorizes rather than generalizes
- No train/validation split
- No batching (single sample training)

---

## Possible Improvements
- Use a larger real-world text corpus
- Add batch training with DataLoader
- Introduce validation set
- Use multi-layer LSTM
- Add dropout for regularization
- Save and reload trained models
- Generate longer sequences with temperature sampling

---

## Conclusion
This project is a minimal end-to-end example of building a custom NLP pipeline with LSTM in PyTorch.
It demonstrates how raw text can be transformed into a supervised learning problem and how sequence models can learn word dependencies.
While simple, it provides a strong foundation for:
- Language modeling
- Text generation
- NLP experimentation with deep learning

