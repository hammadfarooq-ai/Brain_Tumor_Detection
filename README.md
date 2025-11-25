# Brain Tumor Classification using CNN  
**MRI → 4 Classes (Glioma, Meningioma, No Tumor, Pituitary)**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

### Dataset  
[Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
Total images: **3264** → Resized to **150×150**

---

### Final Performance (20 Epochs)

| Metric                | Value      |
|-----------------------|------------|
| **Training Accuracy**   | **95.84%** |
| **Validation Accuracy** | **93.20%** |
| Training Loss         | 0.1314     |
| Validation Loss       | 0.4591     |

#### Training & Validation Accuracy
![Accuracy Plot](Accuracy%20plot.png)

#### Training & Validation Loss
![Loss Plot](loss%20plot.png)

---

### Model Architecture (Custom CNN)
```text
Conv2D → Conv2D → MaxPool → Dropout
→ Conv2D×2 → MaxPool → Dropout
→ Conv2D×3 (128) → MaxPool → Dropout
→ Conv2D(128) → Conv2D(256) → MaxPool → Dropout
→ Flatten → Dense(512)×2 → Dropout → Dense(4, softmax)