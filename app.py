import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Setup Model ---
device = torch.device("cpu") # Hugging face free tier uses CPU
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("breast_model.pth", map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
classes = ["benign", "malignant"]

# Setup Grad-CAM
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# --- UI Layout ---
st.set_page_config(page_title="Explainable Breast Cancer AI", page_icon="🔬")
st.title("🔬 Explainable Breast Cancer AI")
st.write("Upload a histopathology image to see the AI prediction and **Grad-CAM explanation**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    rgb_img = np.array(pil_img.resize((224, 224))) / 255.0
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    st.subheader(f"Prediction: **{classes[pred_idx].upper()}** ({confidence:.1%} confidence)")
    st.image(overlay, caption="Grad-CAM: Red regions most influenced the AI's decision", use_container_width=True)

    st.warning("⚠️ **Medical Disclaimer**: This is an educational research demo, not a clinical diagnostic tool. It should not be used for medical decision-making.")