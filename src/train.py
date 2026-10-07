"""
Transfer-learning classifier for structural damage images.

Dataset:
    PEER Phi-Net Task 2 - Damage State

Classes:
    0 = Damaged
    1 = Undamaged

Expected data directory:
    X_train.npy
    y_train.npy
    X_test.npy
    y_test.npy

Training data:
    4000 images
    85% train = 3400
    15% validation = 600

Official test data:
    1460 images

Example:
    python train.py --model resnet50
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

from torchvision import models
from torchvision.transforms import v2


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = ["Damaged", "Undamaged"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DATASET
# ============================================================

class NpyImageDataset(Dataset):
    """
    Dataset for images stored in NumPy arrays.

    X:
        uint8 images with shape (N, 224, 224, 3)

    y:
        integer labels:
            0 = Damaged
            1 = Undamaged
    """

    def __init__(self, X, y, indices, train=False, img_size=224):

        self.X = X
        self.y = y
        self.indices = indices

        # ImageNet normalization for pretrained torchvision models
        self.normalize = v2.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )

        if train:

            # Mild augmentation.
            # Horizontal flipping is reasonable because structural
            # damage classification should not depend on left/right.
            #
            # RandomResizedCrop introduces moderate variation in
            # scale and framing without aggressive geometric changes.
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),

                v2.RandomResizedCrop(
                    size=img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    antialias=True
                ),
            ])

        else:

            if img_size != 224:

                self.transform = v2.Resize(
                    img_size,
                    antialias=True
                )

            else:

                self.transform = None

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        # Get original dataset index
        dataset_index = self.indices[index]

        # NumPy:
        #     H x W x C
        #
        # PyTorch:
        #     C x H x W
        x = torch.from_numpy(
            self.X[dataset_index]
        ).permute(2, 0, 1)

        # Apply augmentation / resize
        if self.transform is not None:
            x = self.transform(x)

        # Convert uint8 [0,255] -> float [0,1]
        x = x.float() / 255.0

        # ImageNet normalization
        x = self.normalize(x)

        # Convert label to Python integer
        y = int(self.y[dataset_index])

        return x, y


# ============================================================
# MODEL
# ============================================================

def build_model(model_name):

    """
    Load an ImageNet-pretrained model.

    Replace the original classification head with a
    new 2-class classification head.

    Initially the entire backbone is frozen.
    """

    if model_name == "resnet50":

        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

        head = model.fc

    elif model_name == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            2
        )

        head = model.classifier

    elif model_name == "mobilenet_v2":

        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            2
        )

        head = model.classifier

    elif model_name == "inception_v3":

        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )

        # We do not need the auxiliary classifier for this project.
        model.aux_logits = False
        model.AuxLogits = None

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

        head = model.fc

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    # --------------------------------------------------------
    # Freeze entire model
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Unfreeze only classification head
    # --------------------------------------------------------

    for parameter in head.parameters():
        parameter.requires_grad = True

    return model, head


# ============================================================
# FINE-TUNING
# ============================================================

def unfreeze_top(model, model_name):

    """
    Unfreeze the final block(s) of the CNN backbone.

    This allows the pretrained model to adapt its high-level
    features to structural damage images.
    """

    if model_name == "resnet50":

        blocks = [
            model.layer4
        ]

    elif model_name == "efficientnet_b0":

        blocks = [
            model.features[-3:]
        ]

    elif model_name == "mobilenet_v2":

        blocks = [
            model.features[-4:]
        ]

    elif model_name == "inception_v3":

        blocks = [
            model.Mixed_7b,
            model.Mixed_7c
        ]

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    for block in blocks:

        for parameter in block.parameters():
            parameter.requires_grad = True


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None
):

    training = optimizer is not None

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.set_grad_enabled(training):

            # Mixed precision on CUDA
            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            # Backpropagation only during training
            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

        total_loss += (
            loss.item() * len(labels)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += len(labels)

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):

    """
    Generate predictions and probability of class 1
    (Undamaged).
    """

    model.eval()

    predictions = []
    probabilities = []

    for images, _ in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probs = torch.softmax(
            outputs.float(),
            dim=1
        )

        predictions.append(
            probs.argmax(dim=1).cpu()
        )

        # Probability of class 1 = Undamaged
        probabilities.append(
            probs[:, 1].cpu()
        )

    predictions = torch.cat(
        predictions
    ).numpy()

    probabilities = torch.cat(
        probabilities
    ).numpy()

    return predictions, probabilities


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Structural Damage Classification"
    )

    parser.add_argument(
        "--model",
        default="resnet50",
        choices=[
            "resnet50",
            "efficientnet_b0",
            "mobilenet_v2",
            "inception_v3"
        ]
    )

    parser.add_argument(
        "--data_dir",
        default="data"
    )

    parser.add_argument(
        "--out_dir",
        default="results"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Frozen-backbone training epochs"
    )

    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=5,
        help="Fine-tuning epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=None
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # InceptionV3 expects 299x299.
    # Other models use 224x224.
    img_size = (
        args.img_size
        if args.img_size is not None
        else (
            299
            if args.model == "inception_v3"
            else 224
        )
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = Path(
        args.out_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 60)
    print("STRUCTURAL DAMAGE CLASSIFICATION")
    print("=" * 60)

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Model        : {args.model}"
    )

    print(
        f"Image size   : {img_size}"
    )

    print(
        f"Batch size   : {args.batch_size}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    data_dir = Path(
        args.data_dir
    )

    print("\nLoading dataset...")

    X_train = np.load(
        data_dir / "X_train.npy"
    )

    y_train = np.load(
        data_dir / "y_train.npy"
    )

    X_test = np.load(
        data_dir / "X_test.npy"
    )

    y_test = np.load(
        data_dir / "y_test.npy"
    )


    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    print(
        f"X_train shape : {X_train.shape}"
    )

    print(
        f"y_train shape : {y_train.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    print(
        f"y_test shape  : {y_test.shape}"
    )

    print(
        f"X_train dtype : {X_train.dtype}"
    )

    print(
        f"X_test dtype  : {X_test.dtype}"
    )


    # Check labels

    unique_labels = np.unique(
        y_train
    )

    if not np.array_equal(
        unique_labels,
        np.array([0, 1])
    ):

        raise ValueError(
            "Training labels must contain "
            "exactly classes 0 and 1."
        )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    indices = np.arange(
        len(y_train)
    )

    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.val_frac,
        stratify=y_train,
        random_state=args.seed
    )

    print("\nDataset split:")

    print(
        f"Training   : {len(train_indices)}"
    )

    print(
        f"Validation : {len(val_indices)}"
    )

    print(
        f"Test       : {len(y_test)}"
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    def create_loader(
        dataset,
        shuffle
    ):

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda")
        )


    train_dataset = NpyImageDataset(
        X_train,
        y_train,
        train_indices,
        train=True,
        img_size=img_size
    )

    val_dataset = NpyImageDataset(
        X_train,
        y_train,
        val_indices,
        train=False,
        img_size=img_size
    )

    test_dataset = NpyImageDataset(
        X_test,
        y_test,
        np.arange(len(y_test)),
        train=False,
        img_size=img_size
    )


    train_loader = create_loader(
        train_dataset,
        shuffle=True
    )

    val_loader = create_loader(
        val_dataset,
        shuffle=False
    )

    test_loader = create_loader(
        test_dataset,
        shuffle=False
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print("\nLoading pretrained model...")

    model, head = build_model(
        args.model
    )

    model = model.to(device)


    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()


    # ========================================================
    # MIXED PRECISION
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )


    # ========================================================
    # TRAINING PHASES
    # ========================================================

    phases = [
        (
            "head",
            args.epochs,
            args.lr
        )
    ]

    if args.finetune_epochs > 0:

        phases.append(
            (
                "finetune",
                args.finetune_epochs,
                args.finetune_lr
            )
        )


    history = []

    best_val_accuracy = -1.0

    best_state = None

    current_epoch = 0


    # ========================================================
    # TRAINING
    # ========================================================

    for phase, num_epochs, learning_rate in phases:

        # ----------------------------------------------------
        # Fine-tuning phase
        # ----------------------------------------------------

        if phase == "finetune":

            print(
                "\nUnfreezing top layers..."
            )

            unfreeze_top(
                model,
                args.model
            )


        # Get trainable parameters

        trainable_parameters = [
            p
            for p in model.parameters()
            if p.requires_grad
        ]

        number_trainable = sum(
            p.numel()
            for p in trainable_parameters
        )

        print("\n" + "=" * 60)

        print(
            f"PHASE: {phase}"
        )

        print(
            f"Learning rate: {learning_rate}"
        )

        print(
            f"Trainable parameters: "
            f"{number_trainable:,}"
        )

        print("=" * 60)


        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=1e-4
        )


        for epoch in range(
            num_epochs
        ):

            current_epoch += 1

            start_time = time.time()


            # ------------------------------------------------
            # Training mode
            # ------------------------------------------------

            if phase == "head":

                # Keep pretrained backbone in evaluation mode
                # so BatchNorm statistics remain fixed.
                model.eval()

                head.train()

            else:

                model.train()


            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                scaler
            )


            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            model.eval()

            val_loss, val_accuracy = run_epoch(
                model,
                val_loader,
                criterion,
                device
            )


            elapsed = (
                time.time()
                - start_time
            )


            history.append(
                {
                    "epoch": current_epoch,
                    "phase": phase,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                }
            )


            print(
                f"Epoch {current_epoch:2d} "
                f"[{phase:8s}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_accuracy:.4f} "
                f"| Time: {elapsed:.0f}s"
            )


            # ------------------------------------------------
            # Save best validation model
            # ------------------------------------------------

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = (
                    val_accuracy
                )

                best_state = {
                    key: value.detach()
                    .cpu()
                    .clone()

                    for key, value
                    in model.state_dict().items()
                }

                print(
                    "  -> New best validation model"
                )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No best model was saved."
        )

    model.load_state_dict(
        best_state
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        output_dir
        / f"best_{args.model}.pt"
    )

    torch.save(
        best_state,
        model_path
    )


    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "FINAL TEST EVALUATION"
    )

    print("=" * 60)

    predictions, probabilities = predict(
        model,
        test_loader,
        device
    )


    # ========================================================
    # METRICS
    # ========================================================

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )


    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0
    )


    confusion = confusion_matrix(
        y_test,
        predictions
    )


    print(
        f"\nBest validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy:.4f}"
    )


    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )


    print(
        "Confusion matrix "
        "(rows=true, columns=predicted):"
    )

    print(confusion)


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "model": args.model,

        "best_validation_accuracy":
            float(best_val_accuracy),

        "test_accuracy":
            float(test_accuracy),

        "classification_report":
            report,

        "confusion_matrix":
            confusion.tolist(),

        "history":
            history,

        "arguments":
            vars(args)
    }


    metrics_path = (
        output_dir
        / f"{args.model}_metrics.json"
    )


    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )


    # ========================================================
    # SAVE MISCLASSIFIED INDICES
    # ========================================================

    misclassified_indices = np.where(
        predictions != y_test
    )[0]


    np.save(
        output_dir
        / f"{args.model}_misclassified_idx.npy",
        misclassified_indices
    )


    # ========================================================
    # SAVE TEST PROBABILITIES
    # ========================================================

    np.save(
        output_dir
        / f"{args.model}_test_probs.npy",
        probabilities
    )


    # ========================================================
    # CONFUSION MATRIX PLOT
    # ========================================================

    plt.figure(
        figsize=(5, 4)
    )

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        f"{args.model} - Test Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"{args.model}_confusion.png",
        dpi=150
    )

    plt.close()


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    epochs_axis = [
        item["epoch"]
        for item in history
    ]


    train_losses = [
        item["train_loss"]
        for item in history
    ]

    val_losses = [
        item["val_loss"]
        for item in history
    ]

    train_accuracies = [
        item["train_accuracy"]
        for item in history
    ]

    val_accuracies = [
        item["val_accuracy"]
        for item in history
    ]


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )


    # Loss

    axes[0].plot(
        epochs_axis,
        train_losses,
        label="Train"
    )

    axes[0].plot(
        epochs_axis,
        val_losses,
        label="Validation"
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()


    # Accuracy

    axes[1].plot(
        epochs_axis,
        train_accuracies,
        label="Train"
    )

    axes[1].plot(
        epochs_axis,
        val_accuracies,
        label="Validation"
    )

    axes[1].set_title(
        "Training and Validation Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].legend()


    plt.tight_layout()


    plt.savefig(
        output_dir
        / f"{args.model}_curves.png",
        dpi=150
    )

    plt.close()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print("TRAINING COMPLETE")

    print("=" * 60)

    print(
        f"Best validation accuracy : "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Final test accuracy      : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Misclassified test images: "
        f"{len(misclassified_indices)}"
    )

    print(
        f"\nResults saved to: "
        f"{output_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()