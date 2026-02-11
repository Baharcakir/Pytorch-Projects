# Flower Classification with Transfer Learning (MobileNetV2)

This project demonstrates how to apply Transfer Learning using a pretrained MobileNetV2 model in PyTorch to perform multi-class image classification on the Flowers102 dataset.

The goal is to fine-tune a lightweight CNN pretrained on ImageNet to recognize 102 different flower species with minimal training time and computational cost.

## Dataset

Oxford Flowers102 `(from torchvision.datasets)`

- 102 flower classes 
- ~2,040 training images
- ~1,020 validation images
- RGB images with varying resolutions

Loaded via:

```
train_dataset = datasets.Flowers102(root='./data', split="train", transform=transform_train, download=True)
test_dataset  = datasets.Flowers102(root='./data', split="val",   transform=transform_test,  download=True)
```


## Preprocessing & Data Augmentation

Two different pipelines are used:

**Training Transformations**

```
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
```


**Validation Transformations**

```
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
```

**Why?**

- Data augmentation improves generalization.
- Normalization centers pixel values.
- Resizing ensures compatibility with MobileNet input.

## Data Loaders

```
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

---

## Model Architecture

A pretrained MobileNetV2 model is used as the backbone.

Only the final classification layer is replaced.

**Original Classifier**

`Linear(1280 → 1000)`

**Modified Classifier**

`Linear(1280 → 102)`

Implemented as:

```
model = models.mobilenet_v2(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 102)
```

---

## Transfer Learning Strategy
- All pretrained convolutional layers are reused.
- Only the final classifier layer is trained.
- This drastically reduces training time.

---

## Training Setup

| Parameter     | Value                  |
| ------------- | ---------------------- |
| Backbone      | MobileNetV2 (ImageNet) |
| Classes       | 102                    |
| Batch size    | 32                     |
| Optimizer     | Adam                   |
| Learning rate | 0.001                  |
| Loss function | CrossEntropyLoss       |
| LR Scheduler  | StepLR (γ=0.1 every 5) |
| Epochs        | 3                      |
| Device        | CPU / CUDA             |


---

## Training Loop

At each epoch:
- Forward pass
- Compute cross-entropy loss
- Backpropagation
- Update classifier weights
- Step learning rate scheduler

```
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

---

## Model Saving

```
torch.save(model.state_dict(), "mobilenet_flowers102.pth")
```

---

## Evaluation

The model is evaluated on the validation set using:
- Confusion Matrix
- Classification Report
- Overall Accuracy

--- 

## Accuracy

```
Accuracy: 68%
Macro Avg F1-score: 0.66
Weighted Avg F1-score: 0.66
```

---

## Confusion Matrix

Visualized using Seaborn:

```
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, cmap="Blues")
```

This shows how well the model distinguishes between 102 flower categories.

---

## Classification Report

Generated via:

```
print(classification_report(all_labels, all_preds))
```

Metrics include:
- Precision
- Recall
- F1-score
- Support for each class

Some classes are learned well, while others suffer due to:

- Limited samples
- High visual similarity between flowers

---

## Key Concepts Demonstrated

- Transfer Learning
- Fine-tuning pretrained CNNs
- MobileNetV2
- Data augmentation
- Multi-class classification
- Learning rate scheduling
- Confusion matrix analysis
- PyTorch model customization

---

## Limitations

- Only 3 training epochs
- No freezing/unfreezing experiments
- No test-time augmentation
- No hyperparameter tuning
- No class imbalance handling
- No top-k accuracy metrics

---

## Possible Improvements

- Train for more epochs (10–30)
- Freeze backbone first, then unfreeze
- Use stronger models (ResNet50, EfficientNet)
- Add validation accuracy tracking
- Use early stopping
- Apply MixUp or CutMix
- Compute Top-5 accuracy
- Perform Grad-CAM visualization

## How to Run

Install dependencies:
```
pip install torch torchvision matplotlib seaborn scikit-learn tqdm
```

Run the notebook:
```
TransferLearning.ipynb
```

---

## Conclusion

This project shows how powerful transfer learning can be:

With only a few lines of code and minimal training:

- A model pretrained on ImageNet can be adapted
- To solve a 102-class fine-grained vision problem
- Achieving reasonable performance in minutes

This approach is widely used in:

- Medical imaging
- Face recognition
- Plant disease detection

Industrial inspection

- Low-data deep learning scenarios
- A strong foundation for real-world deep learning applications.
