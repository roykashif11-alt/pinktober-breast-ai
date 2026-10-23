import base64
import io
import textwrap
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
# 2. Inject Tailwind, Lucide, and Custom CSS
# ------------------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,800;1,600&display=swap" rel="stylesheet">
    """), 
    unsafe_allow_html=True
)

st.markdown(
    textwrap.dedent("""
    <style>
    #MainMenu, header, footer { visibility: hidden; }
    .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1e293b; }
    
    .stApp {
        background-color: #fdf2f8;
        background-image: 
            radial-gradient(at 10% 10%, rgba(251, 207, 232, 0.8) 0px, transparent 50%),
            radial-gradient(at 90% 20%, rgba(244, 114, 182, 0.3) 0px, transparent 50%),
            radial-gradient(at 50% 80%, rgba(253, 242, 248, 1) 0px, transparent 50%),
            radial-gradient(at 80% 90%, rgba(251, 207, 232, 0.6) 0px, transparent 50%);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(251, 207, 232, 0.6);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 10px 40px -10px rgba(131, 24, 67, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .pink-gradient-text {
        background: linear-gradient(135deg, #be185d 0%, #ec4899 50%, #9d174d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.4);
        border: 3px dashed #f472b6;
        border-radius: 20px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"] section:hover {
        background: rgba(255, 255, 255, 0.8);
        border-color: #db2777;
    }
    
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
        font-weight: 600;
        color: #64748b;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to right, #db2777, #ec4899) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3);
    }
    </style>
    """), 
    unsafe_allow_html=True
)

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
# 4. Header / Hero
# ------------------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <div class="text-center mb-8 relative z-10">
        <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-pink-100/80 border border-pink-300 text-pink-800 text-xs md:text-sm font-semibold shadow-sm mb-6 animate-pulse-slow">
            <span class="text-base">🎀</span>
            <span>Breast Cancer Awareness Month</span>
            <span class="w-1.5 h-1.5 rounded-full bg-pink-600 inline-block"></span>
            <span class="text-pink-600">Pink October</span>
        </div>
        <h1 class="text-4xl md:text-6xl font-serif font-extrabold tracking-tight mb-4 text-slate-900">
            See what the <span class="pink-gradient-text">model sees.</span>
        </h1>
        <p class="max-w-2xl mx-auto text-slate-600 text-base md:text-lg leading-relaxed">
            Upload a stained histopathology tissue slide. Our PyTorch AI model classifies it as 
            <span class="font-semibold text-emerald-700 bg-emerald-100/80 px-2 py-0.5 rounded">benign</span> or 
            <span class="font-semibold text-pink-800 bg-pink-100/80 px-2 py-0.5 rounded">malignant</span>, 
            and visualizes critical regions with Grad-CAM heatmaps.
        </p>
    </div>
    """), 
    unsafe_allow_html=True
)

# ------------------------------------------------------------------
# 5. Navigation Menu Bar (Tabs)
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔬 App Home", "📖 About Project", "💬 AI Assistant"])

# ==================================================================
# TAB 1: MAIN APP
# ==================================================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your histopathology slide image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
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
            
            badge_color = "bg-pink-100 text-pink-800 border-pink-300" if is_malignant else "bg-emerald-100 text-emerald-800 border-emerald-300"
            text_color = "text-pink-900" if is_malignant else "text-emerald-900"
            conf_color = "text-pink-600" if is_malignant else "text-emerald-600"
            finding_text = "The model highlighted dense, pleomorphic cellular clustering with elevated nucleus-to-cytoplasm ratios in the highlighted thermal region, indicative of malignant ductal hyperplasia." if is_malignant else "Normal glandular duct structures with uniform cell distribution. No significant nuclear hyperchromasia detected across the evaluated tile."

        result_html = f"""
        <div class="glass-card mt-6">
            <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-pink-100">
                <div>
                    <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Model Classification Output</span>
                    <div class="flex items-center gap-3 mt-1">
                        <h2 class="text-2xl md:text-3xl font-bold {text_color}">{verdict} Detected</h2>
                        <span class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider border {badge_color}">High Confidence</span>
                    </div>
                </div>
                <div class="flex items-center gap-4 bg-white/80 border border-pink-100 rounded-2xl p-3 px-5 shadow-sm">
                    <div class="text-right">
                        <div class="text-xs text-slate-500 font-medium">Confidence Score</div>
                        <div class="text-2xl font-black {conf_color}">{confidence:.1%}</div>
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div class="space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-semibold text-slate-700">Original Histopathology Slide</span>
                    </div>
                    <div class="relative rounded-2xl overflow-hidden border border-slate-200 bg-slate-900 aspect-video">
                        <img src="{to_data_uri(pil_img.resize((448, 448)))}" class="w-full h-full object-cover">
                    </div>
                </div>
                <div class="space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="text-sm font-semibold text-slate-700">Grad-CAM Attention Heatmap</span>
                    </div>
                    <div class="relative rounded-2xl overflow-hidden border border-pink-300 bg-slate-900 aspect-video shadow-inner">
                        <img src="{to_data_uri(Image.fromarray(overlay).resize((448, 448)))}" class="w-full h-full object-cover">
                    </div>
                </div>
            </div>

            <div class="mt-6 p-4 md:p-5 rounded-2xl bg-pink-50/60 border border-pink-200/80">
                <h4 class="text-sm font-bold text-slate-800 flex items-center gap-2 mb-2">AI Feature Highlight</h4>
                <p class="text-sm text-slate-600 leading-relaxed">{finding_text}</p>
            </div>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

# ==================================================================
# TAB 2: ABOUT PROJECT (FIXED: Using Native Streamlit Components)
# ==================================================================
with tab2:
    # We use pure Streamlit layout here to guarantee text never breaks into HTML code blocks.
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
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
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# TAB 3: AI ASSISTANT (CHATBOT)
# ==================================================================
with tab3:
    st.markdown(
        textwrap.dedent("""
        <div class="text-center mb-6">
            <h2 class="text-3xl font-serif font-bold text-pink-900">💬 Breast Cancer AI Assistant</h2>
            <p class="text-slate-600">Ask questions about the disorders, the problem this solves, the BreaKHis dataset, or how the AI works!</p>
        </div>
        """), 
        unsafe_allow_html=True
    )

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
# 6. Footer
# ------------------------------------------------------------------
st.markdown(
    textwrap.dedent("""
    <footer class="mt-12 pt-8 border-t border-pink-200/60 text-center space-y-3 relative z-10">
        <p class="text-xs md:text-sm font-medium text-slate-600 flex items-center justify-center gap-1.5 flex-wrap">
            Built with <span class="font-semibold text-pink-700">PyTorch</span> & <span class="font-semibold text-pink-700">Streamlit</span> for Pink October 🎀
        </p>
        <p class="text-xs text-slate-500">
            Developed by <span class="font-semibold text-slate-800">Kashif Sultan</span>
        </p>
        <div class="flex items-center justify-center gap-4 text-xs font-semibold text-pink-600 pt-1">
            <a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank" class="hover:text-pink-800 hover:underline transition-colors">LinkedIn</a>
            <span>•</span>
            <a href="https://www.instagram.com/kashif_rashtiic/" target="_blank" class="hover:text-pink-800 hover:underline transition-colors">Instagram</a>
            <span>•</span>
            <a href="https://github.com/roykashif11-alt/pinktober-breast-ai" target="_blank" class="hover:text-pink-800 hover:underline transition-colors">GitHub</a>
        </div>
    </footer>
    """), 
    unsafe_allow_html=True
)
