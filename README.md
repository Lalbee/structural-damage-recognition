# Structural Damage Recognition using Deep Learning

A deep learning project for structural damage recognition from images using binary image classification.

The system classifies structural images into two categories:

- **Damaged**
- **Undamaged**

The project uses transfer learning with pretrained convolutional neural networks (CNNs) and compares multiple architectures to identify the best-performing model.

---

## 1. Problem Statement

Structural damage assessment from images can help automate the identification of damaged and undamaged structures.

The objective of this project is to develop a deep learning-based image classification system that takes a structural image as input and predicts whether the structure is:

- **Damaged (Class 0)**
- **Undamaged (Class 1)**

The project is based on the PEER Hub ImageNet structural damage recognition problem.

---

## 2. Dataset

The project uses the **PEER Hub ImageNet (Φ-Net) – Task 2: Damage State** dataset.

The images are structural images with a resolution of:

```text
224 × 224 × 3