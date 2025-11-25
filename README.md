# Brain Tumor Classification using MRI Images (CNN)

![Brain Tumor MRI](https://img.shields.io/badge/Domain-Medical%20Image%20Classification-blue) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-93.2%25-success)

A Convolutional Neural Network (CNN) model built from scratch to classify brain MRI scans into four categories:

- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**
- **Pituitary Tumor**

---

### Dataset
- **Source**: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) on Kaggle
- **Total Images**: ~3264 (Training + Testing combined)
- **Classes**: 4
- **Image Size**: Resized to `150x150` pixels

---

### Model Performance (After 20 Epochs)

| Metric              | Value      |
|---------------------|------------|
| Training Accuracy   | **95.84%** |
| Validation Accuracy | **93.20%** |
| Training Loss       | 0.1314     |
| Validation Loss     | 0.4591     |

![Training & Validation Accuracy](accuracy_plot.png)  
![Training & Validation Loss](loss_plot.png)

---

### Model Architecture

Custom CNN built using TensorFlow/Keras:

```text
Conv2D(32) → Conv2D(64) → MaxPool → Dropout(0.3)
→ Conv2D(64) ×2 → MaxPool → Dropout(0.3)
→ Conv2D(128) ×3 → MaxPool → Dropout(0.3)
→ Conv2D(128) → Conv2D(256) → MaxPool → Dropout(0.3)
→ Flatten → Dense(512) ×2 → Dropout(0.3) → Dense(4, softmax)