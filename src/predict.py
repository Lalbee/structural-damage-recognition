import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


CLASS_NAMES = {
    0: "Damaged",
    1: "Undamaged"
}


def load_model(model_path, device):

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def predict(model, image_path, device):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    damaged_probability = probabilities[0].item()
    undamaged_probability = probabilities[1].item()

    confidence = probabilities[
        predicted_class
    ].item()

    prediction = CLASS_NAMES[predicted_class]

    print("\nSTRUCTURAL DAMAGE RECOGNITION")
    print("--------------------------------")
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Damaged probability: {damaged_probability * 100:.2f}%")
    print(f"Undamaged probability: {undamaged_probability * 100:.2f}%")

    return prediction, confidence


def main():

    parser = argparse.ArgumentParser(
        description="Predict structural damage from an image."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--model",
        default="results/best_efficientnet_b0.pt",
        help="Path to the trained EfficientNet-B0 model"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    model = load_model(
        model_path,
        device
    )

    predict(
        model,
        image_path,
        device
    )


if __name__ == "__main__":
    main()