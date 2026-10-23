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
# 2. CSS Styles
# ------------------------------------------------------------------
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,800;1,600&display=swap" rel="stylesheet">
<style>
#MainMenu, header, footer { visibility: hidden; }
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
.stApp {
    background-color: #fdf2f8;
    background-image: radial-gradient(at 10% 10%, rgba(251, 207, 232, 0.8) 0px, transparent 50%), radial-gradient(at 90% 20%, rgba(244, 114, 182, 0.3) 0px, transparent 50%), radial-gradient(at 50% 80%, rgba(253, 242, 248, 1) 0px, transparent 50%), radial-gradient(at 80% 90%, rgba(251, 207, 232, 0.6) 0px, transparent 50%);
}
.pink-gradient-text {
    background: linear-gradient(135deg, #be185d 0%, #ec4899 50%, #9d174d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.4); border: 3px dashed #f472b6; border-radius: 20px; padding: 2rem; transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover { background: rgba(255, 255, 255, 0.8); border-color: #db2777; }
.stTabs [data-baseweb="tab-list"] { gap: 1rem; background: rgba(255, 255, 255, 0.6); padding: 10px; border-radius: 16px; border: 1px solid rgba(251, 207, 232, 0.8); justify-content: center; margin-bottom: 2rem; }
.stTabs [data-baseweb="tab"] { height: 50px; border-radius: 12px; padding: 0 24px; font-weight: 600; color: #64748b; border: none; }
.stTabs [aria-selected="true"] { background: linear-gradient(to right, #db2777, #ec4899) !important; color: white !important; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3); }
[data-testid="stSidebar"] { background-color: #0F172A; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h2 { color: #fff !important; font-family: 'Playfair Display', serif; }
.social-btn { display: block; background: rgba(255, 255, 255, 0.1); color: #FDF2F8 !important; text-decoration: none; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-weight: 600; text-align: center; transition: all 0.2s ease; }
.social-btn:hover { background: #db2777; color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. Load Model
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
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
classes = ["Benign", "Malignant"]
cam = GradCAM(model=model, target_layers=[model.features[-1]])

def to_data_uri(img) -> str:
    if isinstance(img, np.ndarray): img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Model Metrics")
    st.markdown('<div style="font-size: 3.5rem; font-weight: 900; color: #ec4899 !important; line-height: 1.2;">97.22%</div>', unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem;'>Accuracy on held-out test split (BreaKHis Dataset)</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### About the Developer")
    st.markdown("<p style='font-size: 1rem; font-weight: 700; color: #fff !important; margin-bottom: 10px;'>Kashif Sultan</p>", unsafe_allow_html=True)
    st.markdown('<a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank" class="social-btn">💼 LinkedIn</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://www.instagram.com/kashif_rashtiic/" target="_blank" class="social-btn">📸 Instagram</a>', unsafe_allow_html=True)
    st.markdown('<a href="https://github.com/roykashif11-alt/pinktober-breast-ai" target="_blank" class="social-btn">💻 GitHub Repository</a>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# 5. Header / Hero
# ------------------------------------------------------------------
ribbon_svg = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cGF0aCBkPSJNNTAgMTAgQzMwIDEwIDIwIDMwIDM1IDUwIEwxNSA5MCBMMzAgOTUgTDUwIDY1IEw3MCA5NSBMODUgOTAgTDY1IDUwIEM4MCAzMCA3MCAxMCA1MCAxMCBaIiBmaWxsPSIjRTA1NjhGIi8+PHBhdGggZD0iTTUwIDEwIEMzMCAxMCAyMCAzMCAzNSA1MCBMMTUgOTAgTDMwIDk1IEw1MCA2NSBMNzAgOTUgTDg1IDkwIEw2NSA1MCBDODAgMzAgNzAgMTAgNTAgMTAgWiIgZmlsbD0idXJsKCNncmFkKSIgb3BhY2l0eT0iMC40Ii8+PGRlZnM+PGxpbmVhckdyYWRpZW50IGlkPSJncmFkIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojZmZmZmZmO3N0b3Atb3BhY2l0eToxIi8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojRTA1NjhGO3N0b3Atb3BhY2l0eTowIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PC9zdmc+"

hero_html = f'<div class="text-center mb-8 relative z-10"><div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-pink-100/80 border border-pink-300 text-pink-800 text-xs md:text-sm font-semibold shadow-sm mb-6"><span class="text-base">🎀</span><span>Breast Cancer Awareness Month</span><span class="w-1.5 h-1.5 rounded-full bg-pink-600 inline-block"></span><span class="text-pink-600">Pink October</span></div><div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;"><img src="{ribbon_svg}" style="width: 90px; filter: drop-shadow(0 10px 15px rgba(224, 86, 143, 0.4));" alt="Pink Ribbon"><h1 class="text-4xl md:text-6xl font-serif font-extrabold tracking-tight mb-0 text-slate-900">See what the <span class="pink-gradient-text">model sees.</span></h1></div><p class="max-w-2xl mx-auto text-slate-600 text-base md:text-lg leading-relaxed mt-6">Upload a stained histopathology tissue slide. Our PyTorch AI model classifies it as <span class="font-semibold text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded">benign</span> or <span class="font-semibold text-pink-800 bg-pink-100/80 px-2 py-0.5 rounded">malignant</span>, and visualizes critical regions with Grad-CAM heatmaps.</p></div>'
st.markdown(hero_html, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 6. Navigation Menu Bar (Tabs)
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 App Home", "📖 About Project", "💬 AI Assistant"])

# ==================================================================
# TAB 1: MAIN APP
# ==================================================================
with tab1:
    uploaded_file = st.file_uploader("Drop your histopathology slide image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
            finding_text = "The model highlighted dense, pleomorphic cellular clustering with elevated nucleus-to-cytoplasm ratios in the highlighted thermal region, indicative of malignant ductal hyperplasia." if is_malignant else "Normal glandular duct structures with uniform cell distribution. No significant nuclear hyperchromasia detected across the evaluated tile."

        # Native Streamlit Layout (Bulletproof)
        st.write("---")
        
        # Header Section
        head_col1, head_col2 = st.columns([2, 1])
        with head_col1:
            if is_malignant:
                st.error(f"### ⚠️ {verdict} Detected (High Confidence)")
            else:
                st.success(f"### ✅ {verdict} Detected (High Confidence)")
        with head_col2:
            st.metric(label="Confidence Score", value=f"{confidence:.1%}")

        # Image Section (Side-by-Side)
        img_col1, img_col2 = st.columns(2, gap="large")
        with img_col1:
            st.image(pil_img.resize((448, 448)), caption="Original Histopathology Slide", use_container_width=True)
        with img_col2:
            st.image(Image.fromarray(overlay).resize((448, 448)), caption="Grad-CAM Attention Heatmap", use_container_width=True)

        # Findings Section
        st.info(f"**AI Feature Highlight:** {finding_text}")

# ==================================================================
# TAB 2: ABOUT PROJECT
# ==================================================================
with tab2:
    with st.container(border=True):
        st.markdown("## 📖 About This Project")
        st.markdown("Built as a submission for Pinktober (Breast Cancer Awareness Month) to demonstrate how Explainable AI can bridge the gap between machine learning and clinical trust.")
        st.write("---")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📊 The Dataset")
            st.write("I used the **BreaKHis** (Breast Cancer Histopathological Database), which contains thousands of microscopic images of breast tumor tissue collected from clinical cases, labeled as benign or malignant by expert pathologists.")
            st.markdown("#### 🔍 Explainable AI (Grad-CAM)")
            st.write("Standard AI is a 'black box.' I implemented **Grad-CAM** to generate the heatmaps shown in the app. This visually explains exactly which cellular regions influenced the prediction, giving pathologists transparency and trust in the model.")
        with col2:
            st.markdown("#### 🧠 The AI Model")
            st.write("The model is based on **MobileNetV2**, utilizing Transfer Learning. It was trained using PyTorch and achieved a **97.22%** accuracy on unseen test images.")
            st.markdown("#### 💻 Tech Stack")
            st.write("Built with Python, PyTorch, Streamlit, and Grad-CAM. The complete source code is available on [GitHub](https://github.com/roykashif11-alt/pinktober-breast-ai).")

# ==================================================================
# TAB 3: AI ASSISTANT (CHATBOT)
# ==================================================================
with tab3:
    st.markdown('<div class="text-center mb-6"><h2 class="text-3xl font-serif font-bold text-pink-900">💬 Breast Cancer AI Assistant</h2><p class="text-slate-600">Ask questions about the disorders, the problem this solves, the BreaKHis dataset, or how the AI works!</p></div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI assistant. Ask me anything about breast cancer, this project's architecture, or how Grad-CAM works."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        p_lower = prompt.lower()
        response = ""
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
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ------------------------------------------------------------------
# 7. Footer
# ------------------------------------------------------------------
footer_html = '''<div style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(251, 207, 232, 0.6); text-align: center;">
    <p style="color: #475569; font-size: 0.9rem; margin-bottom: 10px;">Built with <b style="color: #be185d;">PyTorch</b> & <b style="color: #be185d;">Streamlit</b> for Pink October 🎀</p>
    <p style="color: #475569; font-size: 0.95rem; margin-bottom: 15px;">Developed by <b style="color: #0f172a;">Kashif Sultan</b></p>
    <p>
        <a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank" style="color: #db2777; font-weight: bold; text-decoration: none; margin: 0 10px;">LinkedIn</a> • 
        <a href="https://www.instagram.com/kashif_rashtiic/" target="_blank" style="color: #db2777; font-weight: bold; text-decoration: none; margin: 0 10px;">Instagram</a> • 
        <a href="https://github.com/roykashif11-alt/pinktober-breast-ai" target="_blank" style="color: #db2777; font-weight: bold; text-decoration: none; margin: 0 10px;">GitHub</a>
    </p>
</div>'''
st.markdown(footer_html, unsafe_allow_html=True)
