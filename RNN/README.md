# Time Series Prediction with RNN (PyTorch)

This project implements a **Recurrent Neural Network (RNN)** using **PyTorch** to perform **time series forecasting** on a synthetic sinusoidal dataset.

The goal is to build an end-to-end sequence learning pipeline including:  
- Synthetic data generation  
- Custom sequence dataset creation  
- RNN model definition  
- Training and evaluation  
- Visualization of predictions  

---

## Dataset

**Synthetic Sinusoidal Data**

The dataset is generated using the sine function:

\[
y = \sin(x)
\]

- **Number of samples:** 1000  
- **Sequence length:** 50  
- **Input:** A sliding window of 50 time steps  
- **Target:** The next value in the sequence  

Each training sample looks like:

Input : [y(t), y(t+1), ..., y(t+49)]
Target : y(t+50)


This transforms a continuous signal into a supervised learning problem.

---

## Model Architecture

The model is a simple **vanilla RNN**:

- **RNN layer:**  
  - Input size: 1  
  - Hidden size: 16  
  - Number of layers: 1  
- **Fully connected layer:**  
  - Maps hidden state → output value  

### Forward Pass Logic

- The RNN processes the full sequence.
- Only the **last time step output** is used.
- A linear layer maps it to the final prediction.

---

## Training Details

- **Framework:** PyTorch  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Learning Rate:** 0.001  
- **Batch Size:** 32  
- **Epochs:** 20  

---

## Results

Training loss decreases rapidly, showing that the RNN successfully learns the sinusoidal pattern.

| Epoch | Final Loss |
|-------|------------|
| 1     | 0.1947     |
| 5     | 0.0097     |
| 10    | 0.0012     |
| 20    | **0.0003** |

The model generalizes well to unseen ranges of the sine function.

---

## Testing & Visualization

The trained model is tested on two unseen intervals:

- **Test 1:** sin(100 → 110)  
- **Test 2:** sin(120 → 130)  

For each test:
- The model receives 50 values.
- It predicts the **next time step**.

The final plot shows:
- Training data  
- Test sequences  
- Predicted future values  

This demonstrates the RNN’s ability to **extrapolate temporal patterns**.

---

## How to Run

1. Install dependencies:

```bash
pip install torch torchvision matplotlib numpy jupyter
```
2. Run the script:
```
jupyter notebook RNN.ipynb
```

---

## Key Concepts Demonstrated

- Sequence-to-one prediction
- Sliding window dataset
- RNN hidden states
- Time series forecasting
- PyTorch DataLoader usage

---

## Possible Improvements

- Replace RNN with LSTM or GRU for better long-term memory
- Normalize data for more stable training
- Predict multiple future steps (sequence-to-sequence)
- Add GPU support (.to(device))
- Save and load trained models

---

## Conclusion

This project is a minimal yet complete example of using RNNs for time series prediction in PyTorch.
It demonstrates how sequential data can be transformed into a supervised learning problem and how neural networks can learn temporal dependencies.
