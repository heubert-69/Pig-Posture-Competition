# 🧠 Deep Learning Ensemble for Image Classification

A high-performance deep learning pipeline that combines multiple CNN architectures using a **weighted geometric ensemble (log-probability fusion)** to improve classification accuracy.

This project evaluates and ensembles:
- ResNet18
- MobileNetV3
- EfficientNetB0

---

## 🚀 Key Highlights

- 🔥 Multi-model training (transfer learning)
- 📊 Individual model evaluation
- ⚖️ Accuracy-based weighted ensemble
- 📈 Geometric (log-probability) fusion for stable probability aggregation
- 🧪 Test-time inference pipeline
- ⚡ EfficientNetB0 as strongest base model (~92% accuracy)

---

## 📊 Model Performance

| Model           | Accuracy |
|----------------|----------|
| ResNet18       | 0.851    |
| MobileNetV3    | 0.835    |
| EfficientNetB0 | 0.923    |

---

## 🧠 Ensemble Method

This project uses a **geometric ensemble**, which combines model outputs in log space:


$$\[P_{ensemble} = \exp\left(\sum_{i} w_i \cdot \log(P_i)\right)\]$$

Where:
- $$\( P_i \)$$ = probability distribution from model $i$
- $$\( w_i \)$$ = normalized model weight based on validation accuracy

---

## ⚙️ Pipeline Overview

### 1. Model Training
Each model is fine-tuned using transfer learning on the dataset.

### 2. Evaluation
Validation accuracy is computed for each model:

```python
{'ResNet18': 0.851,
 'MobileNetV3': 0.835,
 'EfficientNetB0': 0.923}
```

3. Weight Computation

Weights are derived from validation performance:

```python
weights = softmax(validation_accuracy * temperature)
```

4. Ensemble Inference
```python
log_probs = 0

for name, probs in models_probs.items():
    weight = ensemble_weights[name]
    log_probs += weight * torch.log(probs + 1e-9)

ensemble_probs = torch.exp(log_probs)
predictions = torch.argmax(ensemble_probs, dim=1)
```

🔬 Key Insights:
EfficientNetB0 dominates performance, but ensembling improves robustness
Geometric averaging stabilizes probability distributions compared to soft voting
Lightweight models still contribute diversity benefits
Proper weight normalization is critical for ensemble stability
---

📈 Future Improvements
 Logit stacking with meta-learner
 Temperature-scaled calibration
 Uncertainty-aware weighting
 Knowledge distillation into single compact model
 Test-time augmentation (TTA) 
--- 
🛠 Tech Stack
PyTorch
Torchvision
NumPy
scikit-learn
CUDA (GPU training)
