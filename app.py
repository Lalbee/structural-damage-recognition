import gdown
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

MODEL_FILE_ID = "1eQRvU2KKOQTzLfhMhUzPyCUh-4SjwNXY"
MODEL_PATH = Path("best_efficientnet_b0.pt")
CLASS_NAMES = {0: "Damaged", 1: "Undamaged"}

def ensure_model():
    if MODEL_PATH.exists():
        return
    print("Downloading trained EfficientNet-B0 model...")
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    result = gdown.download(url, str(MODEL_PATH), quiet=False)
    if not result or not MODEL_PATH.exists():
        raise RuntimeError(
            "Model download failed. Make sure the Google Drive file is "
            "'Anyone with the link - Viewer'."
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensure_model()

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_damage(image):
    if image is None:
        return "Please upload a structural image."

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    damaged = probabilities[0].item()
    undamaged = probabilities[1].item()
    predicted = torch.argmax(probabilities).item()
    prediction = CLASS_NAMES[predicted]
    confidence = probabilities[predicted].item()

    return (
        f"### Prediction: {prediction.upper()}\n\n"
        f"**Confidence:** {confidence * 100:.2f}%\n\n"
        f"**Damaged:** {damaged * 100:.2f}%\n\n"
        f"**Undamaged:** {undamaged * 100:.2f}%"
    )

with gr.Blocks(title="Structural Damage Recognition") as demo:
    gr.Markdown("""
    # 🏗️ Structural Damage Recognition
    ### EfficientNet-B0 based structural damage classification

    Upload an image of a structure and the trained model will
    classify it as **Damaged** or **Undamaged**.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Structural Image"
            )
            predict_button = gr.Button(
                "🔍 Predict Damage",
                variant="primary"
            )

        with gr.Column():
            result_output = gr.Markdown(
                "Upload an image and click **Predict Damage**."
            )

    predict_button.click(
        fn=predict_damage,
        inputs=image_input,
        outputs=result_output
    )

    gr.Markdown("""
    ---
    **Model:** EfficientNet-B0  
    **Classes:** Damaged / Undamaged  
    **Test Accuracy:** 87.26%

    *For demonstration only; not a professional structural safety assessment.*
    """)

if __name__ == "__main__":
    demo.launch(share=True)
