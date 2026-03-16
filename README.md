# Hybrid Skin Cancer Classification using EfficientNet + Swin Transformer

A deep learning project for **skin lesion classification** using a **hybrid architecture that combines EfficientNet-B4 (CNN) and Swin Transformer (Vision Transformer)**. The model is trained on the **HAM10000 dermatoscopic image dataset** and aims to improve classification performance by leveraging both **local feature extraction** and **global contextual understanding**.

---

## Project Overview

Skin cancer is one of the most common cancers worldwide. Early detection through automated image analysis can assist clinicians and researchers in identifying potential cases more efficiently.

This project explores a **hybrid deep learning architecture** that merges the strengths of:

- **Convolutional Neural Networks (CNNs)** for capturing local spatial features
- **Vision Transformers (ViTs)** for modeling global relationships within an image

By combining both approaches, the model learns richer feature representations for **multi-class skin lesion classification**.

---

## Dataset

The model is trained using the **HAM10000 (Human Against Machine with 10000 Training Images)** dataset.

- **Total images:** 10,000+
- **Image type:** Dermatoscopic skin lesion images
- **Number of classes:** 7
- **Source:** Kaggle

Dataset download in the notebook:

```python
import kagglehub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

## Skin Lesion Classes
The HAM10000 dataset contains **7 categories of skin lesions** used for multi-class classification.

| Label | Full Name |
|------|-----------|
| akiec | Actinic keratoses |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

### Class Mapping Used in the Model

```python
lesion_type_dict = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}
## Results

The hybrid EfficientNet + Swin Transformer model achieved strong performance on the HAM10000 dataset.

### Best Validation Accuracy

**89.1%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|---------|
| akiec | 0.61 | 0.82 | 0.70 |
| bcc | 0.75 | 0.93 | 0.83 |
| bkl | 0.90 | 0.68 | 0.78 |
| df | 0.83 | 0.87 | 0.85 |
| mel | 0.79 | 0.67 | 0.73 |
| nv | 0.93 | 0.96 | 0.95 |
| vasc | 1.00 | 0.82 | 0.90 |

**Overall Accuracy:** 0.89

---

### Evaluation Metrics

The model was evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Precision–Recall Curves

These metrics provide a detailed understanding of model performance across all skin lesion classes, especially considering the class imbalance in the dataset.

## Project Status

This project is currently **a work in progress**. I am still experimenting with the model architecture, training strategy, and evaluation methods.

The results shown in this repository are **not the final results**, and I plan to continue improving the model through additional experiments and refinements.

Planned improvements include:

- Further hyperparameter tuning
- Testing alternative architectures
- Improving data augmentation strategies
- Better handling of class imbalance
- Additional evaluation and analysis

