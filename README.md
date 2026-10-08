# Structural Damage Recognition using Deep Learning

A deep learning project for recognizing structural damage from images using binary image classification.

The system classifies structural images into two categories:

- **Damaged**
- **Undamaged**

The project uses transfer learning with multiple pretrained convolutional neural networks (CNNs) and compares their performance to identify the best-performing model for structural damage recognition.

---

## 1. Problem Statement

Structural damage assessment from images can help automate the identification of damaged and undamaged structures.

The objective of this project is to develop a deep learning-based image classification system that takes a structural image as input and predicts whether the structure is:

- **Damaged (Class 0)**
- **Undamaged (Class 1)**

The project is based on the **PEER Hub ImageNet (Φ-Net) – Task 2: Damage State** problem.

---

## 2. Dataset

The project uses the **PEER Hub ImageNet (Φ-Net) – Task 2: Damage State** dataset.

The images are RGB structural images with a resolution of:

```text
224 × 224 × 3
```

### Original Dataset

The original dataset contains:

- **Training images:** 11,811
  - Damaged: 6,282
  - Undamaged: 5,529
- **Test images:** 1,460
  - Damaged: 745
  - Undamaged: 715
- **Total images:** 13,271

### Dataset Used in This Project

Due to computational and storage constraints, a balanced subset of **4,000 training images** was created from the original training data.

The subset contains:

```text
Damaged:     2,000
Undamaged:   2,000
Total:       4,000
```

The subset was created using a fixed random seed of **42** for reproducibility.

The complete **1,460-image test set was kept separate and untouched**.

### Class Mapping

```text
Class 0 → Damaged
Class 1 → Undamaged
```

---

## 3. Data Preparation

The prepared dataset is stored as NumPy arrays for efficient loading during training.

```text
X_train.npy
y_train.npy
X_test.npy
y_test.npy
```

### Train-Validation-Test Split

The 4,000-image training subset was divided using a stratified split:

```text
Training:    3,400 images
Validation:    600 images
Test:        1,460 images
```

The test set was not used during training or model selection.

### Image Preprocessing

All images are processed at:

```text
224 × 224
```

Pixel values are first scaled to `[0, 1]` and then normalized using the ImageNet mean and standard deviation.

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

### Data Augmentation

Data augmentation is applied only to the training set.

The training pipeline uses:

- Random horizontal flip with probability 0.5
- Random resized crop
- Scale range: 0.8–1.0
- Aspect ratio range: 0.9–1.1

Validation and test images are not augmented.

---

## 4. Methodology

The project uses **transfer learning** with pretrained CNN architectures.

Instead of training the complete networks from scratch, pretrained ImageNet models are adapted for binary classification.

### Models Evaluated

The following architectures were evaluated:

1. **ResNet50**
2. **EfficientNet-B0**
3. **MobileNetV2**

The training code also supports **InceptionV3**.

### Training Strategy

Training was performed in two stages.

#### Stage 1 — Classification Head Training

The pretrained backbone was frozen and only the newly added classification head was trained.

Learning rate:

```text
0.001
```

#### Stage 2 — Fine-Tuning

During fine-tuning, the top blocks' weights were unfrozen and trained using a lower learning rate, while the remaining backbone weights stayed frozen.

Learning rate:

```text
0.0001
```

The optimizer used was:

```text
AdamW
```

with:

```text
Weight decay = 1e-4
```

The best checkpoint was selected based on validation accuracy.

---

## 5. Model Comparison

The evaluated models were tested on the untouched 1,460-image test set.

| Model | Best Validation Accuracy | Test Accuracy | Test Errors |
|---|---:|---:|---:|
| ResNet50 | 87.00% | 85.07% | 218 |
| EfficientNet-B0 | 87.83% | **87.26%** | **186** |
| MobileNetV2 | 85.17% | 83.70% | 238 |

### Best Performing Model

Among the models evaluated in our experiments, **EfficientNet-B0 achieved the best performance**.

Test accuracy:

```text
87.26%
```

Correct predictions:

```text
1,274 / 1,460
```

Incorrect predictions:

```text
186 / 1,460
```

---

## 6. EfficientNet-B0 Results

The final EfficientNet-B0 model achieved:

```text
Test Accuracy = 87.26%
```

### Confusion Matrix

The project uses:

```text
Class 0 = Damaged
Class 1 = Undamaged
```

The resulting confusion matrix is:

```text
[[648,  97],
 [ 89, 626]]
```

This represents:

```text
648 Damaged images
→ Correctly classified as Damaged

97 Damaged images
→ Incorrectly classified as Undamaged

89 Undamaged images
→ Incorrectly classified as Damaged

626 Undamaged images
→ Correctly classified as Undamaged
```

Therefore:

```text
Damaged correctly classified:    648 / 745
Undamaged correctly classified:  626 / 715
```

The model missed:

```text
97 damaged images
```

from the 745 damaged images in the test set.

---

## 7. Error Analysis

EfficientNet-B0 produced:

```text
186 total test errors
```

The errors were:

```text
97 Damaged → Undamaged
89 Undamaged → Damaged
```

The model's average confidence was lower on incorrect predictions than on correct predictions.

Some difficult images produced probabilities close to 50–50, indicating uncertainty.

However, some incorrect predictions were also made with high confidence.

Therefore, prediction confidence should not be interpreted as guaranteed correctness.

---

## 8. Inference Pipeline

The project includes a command-line inference script:

```text
src/predict.py
```

The script:

1. Loads the trained EfficientNet-B0 model.
2. Loads an input image.
3. Resizes the image to 224 × 224.
4. Converts it to a tensor.
5. Applies ImageNet normalization.
6. Performs inference.
7. Calculates class probabilities.
8. Displays the predicted class and confidence.

### Prediction Command

From the project root:

```bash
python src/predict.py     --image <path_to_image>     --model <path_to_model>
```

Example:

```bash
python src/predict.py     --image test_image.jpg     --model results/best_efficientnet_b0.pt
```

The model path is optional because the script has the following default:

```text
results/best_efficientnet_b0.pt
```

### Example Output

```text
Device: cuda

STRUCTURAL DAMAGE RECOGNITION
--------------------------------
Image: test_image.jpg
Prediction: DAMAGED
Confidence: 100.00%
Damaged probability: 100.00%
Undamaged probability: 0.00%
```

The script automatically uses CUDA when a compatible GPU is available and otherwise falls back to CPU.

---

## 9. Demo Results

The final inference pipeline was tested successfully using examples from the test set.

### Damaged Example

Test image index:

```text
542
```

Actual class:

```text
Damaged
```

Prediction:

```text
DAMAGED
```

Confidence:

```text
100.00%
```

Probabilities:

```text
Damaged:     100.00%
Undamaged:     0.00%
```

### Undamaged Example

Test image index:

```text
1309
```

Actual class:

```text
Undamaged
```

Prediction:

```text
UNDAMAGED
```

Confidence:

```text
99.99%
```

Probabilities:

```text
Damaged:       0.01%
Undamaged:    99.99%
```

### Borderline Example

Test image index:

```text
0
```

Actual class:

```text
Damaged
```

Prediction:

```text
UNDAMAGED
```

Confidence:

```text
52.69%
```

Probabilities:

```text
Damaged:      47.31%
Undamaged:    52.69%
```

This example demonstrates that visually difficult images can produce uncertain predictions.

---

## 10. Project Structure

```text
structural-damage-recognition/
│
├── data/
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   └── y_test.npy
│
├── results/
│   ├── best_efficientnet_b0.pt
│   ├── efficientnet_b0_confusion.png
│   ├── efficientnet_b0_curves.png
│   ├── efficientnet_b0_metrics.json
│   ├── efficientnet_b0_misclassified_idx.npy
│   └── efficientnet_b0_test_probs.npy
│
├── src/
│   ├── create_subset.py
│   ├── prepare_data.py
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The large dataset files and trained model checkpoint are excluded from GitHub using `.gitignore`.

---

## 11. Installation

Clone the repository:

```bash
git clone https://github.com/Lalbee/structural-damage-recognition.git
cd structural-damage-recognition
```

Create a virtual environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Dataset Setup

The large dataset files are not included in the GitHub repository.

The project expects the processed data in:

```text
data/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

The 4,000-image training subset is created separately in:

```text
data_small/
```

The final subset used for training is placed where the training pipeline expects it:

```text
data/X_train.npy
data/y_train.npy
```

The complete test set remains:

```text
data/X_test.npy
data/y_test.npy
```

### Dataset Files

The following files are intentionally excluded from GitHub because of their size:

```text
X_train.npy
X_test.npy
y_train.npy
y_test.npy
```

The `.gitignore` also contains:

```text
*.npy
```

to prevent accidental upload of NumPy dataset files.

---

## 13. Training

After preparing the dataset, training can be started using:

```bash
python src/train.py
```

The training code supports:

```text
resnet50
efficientnet_b0
mobilenet_v2
inception_v3
```

The training process includes:

1. Dataset loading
2. Stratified train-validation split
3. Data augmentation
4. Classification head training
5. Fine-tuning
6. Validation evaluation
7. Best checkpoint selection
8. Final test evaluation
9. Confusion matrix and metric generation

The best validation checkpoint is saved during training.

---

## 14. Trained Model Checkpoint

The final EfficientNet-B0 checkpoint is:

```text
results/best_efficientnet_b0.pt
```

The checkpoint is excluded from GitHub because `.pt` and `.pth` files are ignored.

Therefore, someone cloning this repository must either:

1. Train the model using `train.py`, or
2. Obtain the trained checkpoint separately.

For the project demonstration, the trained checkpoint is kept locally.

---

## 15. Requirements

The main dependencies are:

```text
torch==2.11.0
torchvision==0.26.0
numpy
scikit-learn
matplotlib
seaborn
pillow
```

The project was tested using:

```text
PyTorch: 2.11.0+cu130
Torchvision: 0.26.0+cu130
GPU: Tesla T4
CUDA: Available
```

The exact CUDA build may depend on the environment.

---

## 16. Reproducibility

A fixed random seed of:

```text
42
```

was used for dataset subset selection and train-validation splitting.

The training subset contains equal numbers of both classes:

```text
2,000 Damaged
2,000 Undamaged
```

The complete 1,460-image test set was kept separate from training and validation.

The best checkpoint was selected based on validation accuracy rather than test accuracy.

---

## 17. Limitations

The current project has several limitations:

- Only a 4,000-image subset of the available training images was used.
- The model was trained on the PEER structural image dataset.
- Performance on arbitrary real-world photographs may differ because of domain shift.
- Some visually ambiguous images are misclassified.
- The final test accuracy is 87.26%, so the model does not correctly classify every image.
- High prediction confidence does not guarantee correctness.
- The current inference pipeline resizes input images directly to 224 × 224.
- The system performs binary image classification and does not identify the exact type, severity, or location of structural damage.

Therefore, the system should be considered a **deep learning-based structural damage classification prototype** and not a replacement for professional structural inspection.

---

## 18. Conclusion

This project developed a deep learning-based system for binary structural damage recognition from images.

Three transfer-learning architectures were evaluated:

- **ResNet50**
- **EfficientNet-B0**
- **MobileNetV2**

Among the models evaluated in our experiments, **EfficientNet-B0 achieved the best test accuracy of 87.26%** on the untouched 1,460-image test set.

The final system includes:

- Balanced training subset creation
- Image preprocessing
- Data augmentation
- Transfer learning
- Two-stage training
- Multiple CNN model comparison
- Validation-based model selection
- Test-set evaluation
- Confusion matrix and error analysis
- Command-line inference using `predict.py`
- Class probability output for prediction interpretation

The project demonstrates the practical use of transfer learning for automated structural damage recognition while also highlighting the challenges of ambiguous images, model confidence, and domain shift.

---

## 19. GitHub Notes

Large datasets and trained model checkpoints are intentionally excluded from the repository.

The `.gitignore` contains:

```text
data/
data_small/
*.npy
*.pt
*.pth
__pycache__/
*.pyc
```

This prevents large dataset and model files from being accidentally committed to GitHub.

The repository therefore contains the source code, configuration files, documentation, and selected result files required to understand and reproduce the project.

---

## 20. Final Project Summary

```text
Task:
Structural Damage Recognition

Problem Type:
Binary Image Classification

Dataset:
PEER Hub ImageNet (Φ-Net) – Task 2: Damage State

Training Subset:
4,000 images

Training:
3,400 images

Validation:
600 images

Test:
1,460 images

Models Evaluated:
ResNet50
EfficientNet-B0
MobileNetV2

Best Model:
EfficientNet-B0

Test Accuracy:
87.26%

Test Errors:
186 / 1,460

Damaged Class:
0

Undamaged Class:
1

Inference:
src/predict.py
```
