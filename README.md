# Multimodal Pet Photo Engagement Prediction
### A CNN-Metadata Fusion Approach to Pawpularity Score Prediction
**DS440 Capstone Project — Pennsylvania State University, Spring 2026**

---

## Overview

This project investigates whether combining deep image features with structured metadata can improve prediction of pet photo engagement scores on the [PetFinder.my](https://www.petfinder.my/) platform.

Each pet photo on PetFinder receives a **Pawpularity score** (0–100) derived from real user engagement data — page visits and interaction rates. A model that can predict this score before a photo is posted could help shelters identify low-engagement photos and retake them, potentially improving adoption rates.

**Research question:** Can a late-fusion multimodal architecture combining ResNet-18 image features with binary metadata features outperform an image-only baseline for Pawpularity score prediction?

---

## Architecture

The model is a late-fusion multimodal neural network defined as:

```
ŷ = h( [f(x₁) ; g(x₂)] )
```

Where:
- **x₁ ∈ ℝ^(3×160×160)** — input image tensor
- **x₂ ∈ {0,1}¹²** — binary metadata vector (12 photo characteristics)
- **f** — image encoder (ResNet-18, pretrained on ImageNet) → **z₁ ∈ ℝ⁵¹²**
- **g** — metadata encoder (MLP) → **z₂ ∈ ℝ¹⁶**
- **h** — fusion regression head → scalar Pawpularity prediction

### Image Encoder f
ResNet-18 pretrained on ImageNet with the classification head replaced by an identity layer, outputting a 512-dimensional feature vector:

```
z₁ = f(x₁),   z₁ ∈ ℝ⁵¹²
```

Each residual block computes **F(x) + x**, where the skip connection prevents vanishing gradients during backpropagation.

### Metadata Encoder g
A single-layer MLP with ReLU activation and dropout:

```
z₂ = g(x₂) = Dropout(ReLU(W₁x₂ + b₁)),   W₁ ∈ ℝ^(16×12)
```

ReLU is necessary because Pearson correlation analysis revealed near-zero individual correlation between all metadata features and Pawpularity (range: −0.024 to 0.017), meaning a linear encoder would contribute negligible signal.

### Fusion Head h
Image and metadata representations are concatenated and passed through:

```
z = [z₁ ; z₂] ∈ ℝ⁵²⁸

ŷ = h(z) = W₃(Dropout(ReLU(W₂z + b₂))) + b₃
```

Where **W₂ ∈ ℝ^(64×528)** and **W₃ ∈ ℝ^(1×64)**.

---

## Dataset

- **Source:** [PetFinder.my Pawpularity Score — Kaggle Competition (2021)](https://www.kaggle.com/competitions/petfinder-pawpularity-score)
- **Size:** 9,912 training images of cats and dogs
- **Metadata:** 12 binary features (Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur)
- **Target:** Pawpularity score (0–100), derived from platform engagement data

---

## Key Findings

### Model Performance
| Metric | Value |
|--------|-------|
| Mean CV RMSE | **20.09** |
| Best score range RMSE | **11.53** (scores 21–40) |
| Worst score range RMSE | **48.57** (scores 81–100) |
| OOF Mean Absolute Error | **14.88** |

An RMSE of 20.09 places the model within the range of competitive Kaggle leaderboard submissions (17.0–20.0).

### Overfitting
Training loss dropped **77%** across 5 epochs while validation loss dropped only **9%**, indicating significant overfitting — expected given only 10,000 training images for an 11M parameter model.

### Spurious Open Mouth Correlation
Manual inspection of the top 50 highest-confidence predictions revealed a key failure mode: the model systematically overpredicts scores for photos where the pet's mouth is open.

| | Top 50 Predictions | Full Dataset |
|---|---|---|
| Face visible | 100% | 90.4% |
| Eyes visible | 90% | 77.3% |
| Blurry | 4% | 7.0% |
| True mean score | 72.3 | 38.0 |
| Predicted mean score | 88.5 | 37.0 |
| Mean absolute error | 24.31 | 14.88 |

The model learned that expressive, front-facing pets tend to score highly — but open mouths do not causally drive engagement. This is a spurious correlation caused by insufficient training data to learn counterexamples.

### Metadata Sensitivity
Across all 12 metadata features, no feature shifted mean predictions by more than **0.73 points**, suggesting the metadata branch contributed negligible signal relative to the image pathway. A future image-only ablation is needed to confirm this quantitatively.

---

## Metadata Correlation Analysis

Pearson correlation between each metadata feature and Pawpularity score:

| Feature | Pearson r | \|r\| |
|---------|-----------|-------|
| Blur | −0.0235 | 0.0235 |
| Group | +0.0165 | 0.0165 |
| Accessory | +0.0133 | 0.0133 |
| Subject Focus | −0.0099 | 0.0099 |
| Face | +0.0080 | 0.0080 |
| Eyes | −0.0067 | 0.0067 |
| Info | −0.0047 | 0.0047 |
| Human | +0.0040 | 0.0040 |
| Occlusion | +0.0020 | 0.0020 |
| Collage | +0.0017 | 0.0017 |
| Action | −0.0014 | 0.0014 |
| Near | +0.0010 | 0.0010 |

All features show near-zero individual predictive power, motivating the fusion design to capture non-linear feature interactions.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Loss function | MSELoss |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻⁴ |
| Weight decay | 1×10⁻⁴ |
| LR scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Image size | 160 × 160 px |
| Epochs | 5 |
| Precision | bfloat16 AMP |
| Normalization | μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225] |
| Cross-validation | 5-fold stratified |

---

## Repository Structure

```
pawpularity-prediction/
├── README.md
├── notebooks/
│   ├── pawpularity_project.ipynb       # Main project code (ResNet-18 + metadata fusion)
│   └── Swin_T__No_Mixup_.ipynb         # Swin-T reference baseline (no Mixup)
├── results/
│   ├── oof_predictions.csv             # Out-of-fold predictions across all 9,912 samples
│   ├── metadata_importance.csv         # Pearson correlation results
│   └── training_history.csv           # Per-epoch train/val loss across all folds
└── report/
└── DS_440_Capstone.pdf             # Final research paper
```

---

## Requirements

```bash
pip install torch torchvision timm pandas numpy matplotlib seaborn scikit-learn
```

**Hardware:** GPU strongly recommended. Trained on Google Colab T4 GPU (~1.5 hours total for 5-fold CV).

---

## How to Run

**1. Mount your Google Drive and set the data path:**
```python
config.root = "/content/drive/MyDrive"
```

**2. Run the full training pipeline:**
```bash
# Open pawpularity_project.ipynb in Google Colab
# Set runtime to GPU (Runtime > Change runtime type > T4 GPU)
# Run all cells
```

**3. Results are saved to:**
```
outputs/oof_predictions.csv
outputs/training_history.csv
```

---

## Limitations

- No image-only ablation baseline — cannot confirm whether metadata fusion improved performance
- 10,000 training images is small for deep learning, causing overfitting and spurious correlations
- Binary metadata loses important gradations (e.g., slightly blurry vs. completely blurry)
- Model fails at extreme score ranges — RMSE 48.57 for scores 81–100

---

## Future Work

- Run image-only ablation (code provided in `pawpularity_project.ipynb` as `ImageOnlyModel`)
- Replace binary metadata with continuous measurements (blurriness score, face detection confidence)
- Apply stronger augmentation and higher dropout to address overfitting
- Test EfficientNet-B0 backbone as a stronger but memory-efficient alternative
- Ensemble OOF predictions across multiple model architectures

---

## References

- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR*.
- Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
- PetFinder.my Pawpularity Score. Kaggle Competition, 2021.
