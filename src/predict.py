import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "results/best_efficientnet_b0.pt"

CLASS_NAMES = {
    0: "Damaged",
    1: "Undamaged"
}


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Load EfficientNet-B0
# ------------------------------------------------------------
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

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

    print("\n" + "=" * 55)
    print("STRUCTURAL DAMAGE CLASSIFICATION")
    print("=" * 55)

    print(f"Image                 : {image_path}")
    print(f"Prediction            : {prediction.upper()}")
    print(f"Confidence            : {confidence * 100:.2f}%")

    print("-" * 55)

    print(
        f"Damaged probability   : "
        f"{damaged_probability * 100:.2f}%"
    )

    print(
        f"Undamaged probability : "
        f"{undamaged_probability * 100:.2f}%"
    )

    print("=" * 55)

    return prediction, confidence


# ------------------------------------------------------------
# Command-line usage
# ------------------------------------------------------------
if __name__ == "__main__":

    image_path = input(
        "\nEnter image path: "
    ).strip()

    predict(image_path)