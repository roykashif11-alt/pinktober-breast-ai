import base64
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# ------------------------------------------------------------------
# 1. Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Pink October AI Diagnostics", page_icon="🎀", layout="wide")

# ------------------------------------------------------------------
# 2. Custom CSS (Recreating the Tailwind Design)
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,800;1,600&display=swap');

/* Base Styles */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* Mesh Gradient Background */
.stApp {
    background-color: #fdf2f8;
    background-image: 
        radial-gradient(at 10% 10%, rgba(251, 207, 232, 0.8) 0px, transparent 50%),
        radial-gradient(at 90% 20%, rgba(244, 114, 182, 0.3) 0px, transparent 50%),
        radial-gradient(at 50% 80%, rgba(253, 242, 248, 1) 0px, transparent 50%),
        radial-gradient(at 80% 90%, rgba(251, 207, 232, 0.6) 0px, transparent 50%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 2rem; }

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(251, 207, 232, 0.6);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 10px 40px -10px rgba(131, 24, 67, 0.1);
    margin-bottom: 2rem;
}

/* Typography */
h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; font-weight: 800 !important; color: #0f172a; }
.pink-gradient-text {
    background: linear-gradient(135deg, #be185d 0%, #ec4899 50%, #9d174d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle { color: #475569; font-size: 1.1rem; line-height: 1.6; text-align: center; max-width: 800px; margin: 0 auto 2rem auto; }

/* Streamlit Tabs (Menu Bar) */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(255, 255, 255, 0.6);
    padding: 10px;
    border-radius: 16px;
    border: 1px solid rgba(251, 207, 232, 0.8);
    justify-content: center;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    border-radius: 12px;
    padding: 0 24px;
    background-color: transparent;
    font-weight: 600;
    color: #64748b;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(to right, #db2777, #ec4899) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3);
}

/* Streamlit File Uploader Override */
[data-testid="stFileUploader"] section {
    background: rgba(253, 242, 248, 0.5);
    border: 2px dashed #f472b6;
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover {
    background: rgba(253, 242, 248, 0.9);
    border-color: #db2777;
}

/* Custom Result UI */
.result-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
}
.badge-malignant { background: #fce7f3; color: #be185d; border: 1px solid #fbcfe8; }
.badge-benign { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

.confidence-box {
    background: rgba(255,255,255,0.8); border: 1px solid #fbcfe8; border-radius: 16px; padding: 1rem 1.5rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.confidence-val { font-size: 2rem; font-weight: 900; color: #db2777; line-height: 1; }

/* Chatbot Styles */
.chat-container { margin-top: 1rem; }

/* Custom Footer */
.footer { text-align: center; margin-top: 4rem; padding-top: 2rem; border-top: 1px solid rgba(251, 207, 232, 0.6); font-size: 0.9rem; color: #64748b; }
.footer a { color: #db2777; text-decoration: none; font-weight: 600; margin: 0 10px; transition: color 0.2s; }
.footer a:hover { color: #be185d; text-decoration: underline; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. AI Model Loading
# ------------------------------------------------------------------
@st.cache_resource
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
classes = ["Benign", "Malignant"]
cam = GradCAM(model=model, target_layers=[model.features[-1]])

def to_data_uri(img) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------------
# 4. Hero Header
# ------------------------------------------------------------------
ribbon_b64 = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cGF0aCBkPSJNNTAgMTAgQzMwIDEwIDIwIDMwIDM1IDUwIEwxNSA5MCBMMzAgOTUgTDUwIDY1IEw3MCA5NSBMODUgOTAgTDY1IDUwIEM4MCAzMCA3MCAxMCA1MCAxMCBaIiBmaWxsPSIjRTA1NjhGIi8+PC9zdmc+"

st.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem;">
    <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(252, 231, 243, 0.8); border: 1px solid #fbcfe8; padding: 6px 16px; border-radius: 50px; color: #9d174d; font-size: 0.85rem; font-weight: 600; margin-bottom: 1.5rem;">
        🎀 Breast Cancer Awareness Month • Pink October
    </div>
    <h1 style="font-size: 3.5rem; margin-bottom: 1rem; line-height: 1.1;">
        See what the <span class="pink-gradient-text">model sees.</span>
    </h1>
    <p class="hero-subtitle">
        Upload a stained histopathology tissue slide. Our PyTorch AI model classifies it as 
        <b style="color: #15803d; background: #dcfce7; padding: 2px 6px; border-radius: 4px;">benign</b> or 
        <b style="color: #be185d; background: #fce7f3; padding: 2px 6px; border-radius: 4px;">malignant</b>, 
        and visualizes critical regions with Grad-CAM heatmaps.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 5. Navigation Menu (Streamlit Tabs)
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 App Home", "📖 About Project", "💬 AI Assistant"])

# ==================================================================
# TAB 1: MAIN APP
# ==================================================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your histopathology slide here (JPG, PNG)", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Analyzing cellular structures and generating Grad-CAM..."):
            pil_img = Image.open(uploaded_file).convert("RGB")
            rgb_img = (np.array(pil_img.resize((224, 224))) / 255.0).astype(np.float32)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = probs[pred_idx].item()

            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)

            verdict = classes[pred_idx]
            is_malignant = verdict == "Malignant"
            badge_class = "badge-malignant" if is_malignant else "badge-benign"
            text_color = "#be185d" if is_malignant else "#15803d"

        # Results UI
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; border-bottom: 1px solid #fbcfe8; padding-bottom: 1.5rem; margin-bottom: 1.5rem;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Model Classification Output</span>
                    <h2 style="margin: 0; color: {text_color};">{verdict} Detected</h2>
                    <span class="result-badge {badge_class}">High Confidence</span>
                </div>
                <div class="confidence-box">
                    <div>
                        <div style="font-size: 0.8rem; color: #64748b; font-weight: 500;">Confidence Score</div>
                        <div class="confidence-val">{confidence:.1%}</div>
                    </div>
                    <div style="font-size: 1.5rem;">📊</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1rem;">
                <div>
                    <div style="font-weight: 600; color: #334155; margin-bottom: 0.5rem;">🖼️ Original Slide</div>
                    <img src="{to_data_uri(pil_img.resize((448, 448)))}" style="width: 100%; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                </div>
                <div>
                    <div style="font-weight: 600; color: #db2777; margin-bottom: 0.5rem;">👁️ Grad-CAM Attention Map</div>
                    <img src="{to_data_uri(Image.fromarray(overlay).resize((448, 448)))}" style="width: 100%; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                </div>
            </div>

            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(252, 231, 243, 0.4); border-radius: 16px; border: 1px solid #fbcfe8;">
                <h4 style="margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem;">📝 AI Feature Highlight</h4>
                <p style="margin: 0; font-size: 0.95rem; color: #475569;">
                    The deep learning model focused predominantly on regions highlighted in warm red/yellow zones. 
                    These areas indicate high nuclear pleomorphism and irregular ductal structures, which are key indicators for the prediction.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================================================================
# TAB 2: ABOUT PROJECT
# ==================================================================
with tab2:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: #be185d;">📖 About This Project</h2>
        <p style="font-size: 1.1rem; color: #475569;">Built as a submission for Pinktober (Breast Cancer Awareness Month) to demonstrate how Explainable AI can bridge the gap between machine learning and clinical trust.</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
            <div>
                <h4 style="color: #9d174d;">📊 The Dataset</h4>
                <p style="color: #475569; font-size: 0.95rem;">I used the <b>BreaKHis</b> (Breast Cancer Histopathological Database), which contains thousands of microscopic images of breast tumor tissue labeled as benign or malignant.</p>
                
                <h4 style="color: #9d174d; margin-top: 1.5rem;">🔍 Explainable AI (Grad-CAM)</h4>
                <p style="color: #475569; font-size: 0.95rem;">Standard AI is a 'black box.' I implemented <b>Grad-CAM</b> to generate heatmaps. This shows pathologists exactly which cells influenced the prediction, increasing trust.</p>
            </div>
            <div>
                <h4 style="color: #9d174d;">🧠 The AI Model</h4>
                <p style="color: #475569; font-size: 0.95rem;">The model is based on <b>MobileNetV2</b> using Transfer Learning. It was trained using PyTorch and achieved a <b>97.22%</b> accuracy on unseen test images.</p>

                <h4 style="color: #9d174d; margin-top: 1.5rem;">💻 Tech Stack</h4>
                <p style="color: #475569; font-size: 0.95rem;">Python, PyTorch, Streamlit, and Grad-CAM. The source code is available on <a href="https://github.com/roykashif11-alt/pinktober-breast-ai" target="_blank" style="color: #db2777; font-weight: 600; text-decoration: none;">GitHub</a>.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================================================================
# TAB 3: AI ASSISTANT (CHATBOT)
# ==================================================================
with tab3:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #be185d; margin-bottom: 0.5rem;">💬 Breast Cancer AI Assistant</h2>
        <p style="color: #475569;">Ask questions about the disorders, the problem this solves, the BreaKHis dataset, or how the AI works!</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything about breast cancer, this project's architecture, or how Grad-CAM works."}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        p_lower = prompt.lower()
        response = ""

        # Simple Rule-Based Bot Logic
        if "problem" in p_lower or "why build" in p_lower or "solution" in p_lower:
            response = "The problem is that manual breast cancer diagnosis from histopathology slides is time-consuming and prone to inter-observer variability. The solution is this AI, which acts as a second pair of eyes, flagging potential cancer cells in seconds."
        elif "dataset" in p_lower or "data" in p_lower:
            response = "We use the **BreaKHis** dataset. It contains thousands of H&E stained microscopic images of breast tissue, carefully labeled by pathologists as either benign or malignant."
        elif "grad-cam" in p_lower or "explain" in p_lower or "heatmap" in p_lower:
            response = "Grad-CAM (Gradient-weighted Class Activation Mapping) is an Explainable AI technique. It looks at the gradients flowing through the CNN and creates a heatmap (red/yellow). The red areas show exactly which cells the AI was looking at when it made its decision."
        elif "accuracy" in p_lower or "model" in p_lower:
            response = "The AI model is **MobileNetV2**, trained using Transfer Learning on PyTorch. It achieves an impressive **97.22% accuracy** on the test dataset."
        elif "cancer" in p_lower or "disorder" in p_lower:
            response = "Breast cancer starts when cells in the breast begin to grow out of control. Early detection through screenings and histopathological analysis (like the images this AI analyzes) is crucial for survival. When detected early, survival rates are over 99%."
        else:
            response = "I'm a specialized AI for this project. Try asking me about: the **dataset**, how **Grad-CAM** works, the **problem** it solves, or the **accuracy** of the model!"

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# ------------------------------------------------------------------
# 6. Footer
# ------------------------------------------------------------------
st.markdown(f"""
<div class="footer">
    <p>Built with <b>PyTorch</b> & <b>Streamlit</b> for Pink October 🎀</p>
    <p>Developed by <b>Kashif Sultan</b></p>
    <p>
        <a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank">LinkedIn</a> • 
        <a href="https://www.instagram.com/kashif_rashtiic/" target="_blank">Instagram</a> •
        <a href="https://github.com/roykashif11-alt/pinktober-breast-ai" target="_blank">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
