import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import pandas as pd

# --- 1. Page Configuration ---
st.set_page_config(page_title="Pinktober AI Diagnostics", page_icon="🎀", layout="wide")

# --- 2. Setup Model ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model():
    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("breast_model.pth", map_location=device))
    model.to(device).eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
classes = ["Benign", "Malignant"]

target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# --- 3. Sidebar Information ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Breast_Cancer_Awareness_Ribbon.svg/200px-Breast_Cancer_Awareness_Ribbon.svg.png", width=100)
    st.header("About This Project")
    st.info(
        "This is an **Explainable AI (XAI)** tool for breast cancer histopathology. "
        "It uses a fine-tuned **MobileNetV2** Convolutional Neural Network trained on the BreaKHis dataset."
    )
    st.metric(label="Model Test Accuracy", value="97.22%")
    st.markdown("---")
    st.subheader("How it works:")
    st.markdown(
        "1. Upload a microscope slide image.\n"
        "2. The AI predicts Benign or Malignant.\n"
        "3. **Grad-CAM** generates a heatmap showing exactly which cells influenced the AI's decision."
    )

# --- 4. Main UI Layout ---
st.title("🎀 Explainable Breast Cancer AI")
st.markdown("Upload a histopathology image to see the AI prediction and Grad-CAM explanation.")

uploaded_file = st.file_uploader("Choose an image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display a spinner while processing
    with st.spinner("Analyzing image..."):
        # Load image
        pil_img = Image.open(uploaded_file).convert("RGB")
        
        # Preprocess
        rgb_img = np.array(pil_img.resize((224, 224))) / 255.0
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        # Grad-CAM
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # --- 5. Display Results in Columns ---
    st.markdown("---")
    
    # Prediction Header
    if classes[pred_idx] == "Malignant":
        st.error(f"### Prediction: **{classes[pred_idx].upper()}** ({confidence:.1%} confidence)")
    else:
        st.success(f"### Prediction: **{classes[pred_idx].upper()}** ({confidence:.1%} confidence)")

    # Image Columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded Original Image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM: Red regions influenced the AI", use_container_width=True)

    # Probability Chart
    st.markdown("### AI Probability Breakdown")
    prob_df = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": [probs[0].item(), probs[1].item()]
    })
    st.bar_chart(prob_df.set_index("Class"), color="#db2777")

    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **Medical Disclaimer**: This is an educational research demo, not a clinical diagnostic tool. It should not be used for medical decision-making. Please consult a qualified pathologist.")
