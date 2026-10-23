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
st.set_page_config(page_title="Slide Reader · Explainable AI", page_icon="🎀", layout="wide")

# ------------------------------------------------------------------
# 2. Premium Design System
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap');

:root{
  --ink:#0F172A;          /* Dark slate for text */
  --glass:rgba(255, 255, 255, 0.7);
  --eosin:#E0568F;        /* Pink accent */
  --hema:#4C1D95;         /* Deep purple accent */
  --benign:#0EA5E9;
  --malignant:#E11D48;
}

html, body, [class*="css"], .stApp{
  font-family:'Inter',sans-serif;
  color:var(--ink);
}

/* Premium Mesh Gradient Background */
.stApp{
  background-color: #FDF2F8;
  background-image: 
    radial-gradient(at 0% 0%, hsla(253,16%,7%,0) 0, transparent 50%), 
    radial-gradient(at 50% 0%, hsla(225,39%,30%,0) 0, transparent 50%), 
    radial-gradient(at 100% 0%, hsla(339,49%,30%,0) 0, transparent 50%),
    radial-gradient(at 0% 50%, #FCE7F3 0, transparent 50%),
    radial-gradient(at 100% 50%, #E0E7FF 0, transparent 50%),
    radial-gradient(at 50% 100%, #F3E8FF 0, transparent 50%);
  background-attachment: fixed;
}

#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:2rem; max-width:1100px;}

h1,h2,h3,.serif{font-family:'Outfit',sans-serif !important; color:var(--ink); font-weight: 800;}

/* ---------- Glass Hero Card ---------- */
.hero-card {
  background: var(--glass);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.8);
  border-radius: 24px;
  padding: 3rem;
  box-shadow: 0 10px 40px -10px rgba(76, 29, 149, 0.1);
  margin-bottom: 2rem;
  display: flex;
  align-items: center;
  gap: 2rem;
}
.hero-text h1 {
  font-size: clamp(2.5rem, 5vw, 4rem);
  line-height: 1.1;
  margin: 0 0 1rem 0;
  background: linear-gradient(135deg, var(--ink) 0%, var(--hema) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-text p {
  font-size: 1.15rem;
  color: #475569;
  max-width: 55ch;
  line-height: 1.6;
  margin: 0;
}
.ribbon-svg {
  width: 120px;
  min-width: 120px;
  filter: drop-shadow(0 10px 10px rgba(224, 86, 143, 0.3));
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] section{
  background: rgba(255, 255, 255, 0.6);
  border: 2px dashed var(--eosin);
  border-radius: 20px; 
  padding: 2rem;
  transition: all 0.3s ease;
}
[data-testid="stFileUploader"] section:hover{
  background: #fff;
  border-color: var(--hema);
  box-shadow: 0 4px 20px rgba(224, 86, 143, 0.15);
}
[data-testid="stFileUploader"] button{
  background: var(--hema); 
  color:#fff; 
  border:none; 
  border-radius:10px;
  padding: 0.5rem 1.5rem;
  font-weight: 600;
}

/* ---------- Empty State Steps ---------- */
.steps-container {
  display: flex; gap: 1.5rem; margin-top: 3rem; justify-content: center;
}
.step-card {
  flex: 1;
  background: rgba(255, 255, 255, 0.5);
  border-radius: 16px;
  padding: 1.5rem;
  border: 1px solid #fff;
  text-align: center;
}
.step-num {
  background: var(--eosin); color: white; width: 32px; height: 32px;
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: bold; margin: 0 auto 1rem auto;
}
.step-card h4 { margin: 0 0 0.5rem 0; font-family: 'Outfit'; font-size: 1.2rem; }
.step-card p { font-size: 0.9rem; color: #64748B; margin: 0; line-height: 1.4; }

/* ---------- Verdict ---------- */
.verdict{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;
  border-radius:20px; padding:2rem; color:#fff; margin:1rem 0 2rem;
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2);
}
.verdict.malignant{background:linear-gradient(135deg, #E11D48, #9F1239);}
.verdict.benign{background:linear-gradient(135deg, #0EA5E9, #0369A1);}
.verdict .label{font-size:1rem; opacity:.9; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;}
.verdict .call{font-family:'Outfit',sans-serif; font-size:clamp(2.5rem,5vw,4rem); font-weight:800; line-height:1;}
.verdict .conf{text-align:right;}
.verdict .conf b{font-family:'Outfit',sans-serif; font-size:clamp(2.5rem,5vw,4rem); font-weight:600; display:block; line-height:1;}

/* ---------- Field of View ---------- */
.fov-wrap{text-align:center;}
.fov{
  width:100%; aspect-ratio:1/1; border-radius:50%; overflow:hidden;
  border:8px solid #fff;
  box-shadow: 0 0 0 2px #E2E8F0, 0 20px 40px rgba(15, 23, 42, 0.15);
  background:#000;
}
.fov img{width:100%; height:100%; object-fit:cover; display:block;}
.fov-cap{margin-top:1.2rem; font-size:.95rem; color:#475569;}
.fov-cap b{font-family:'Outfit',sans-serif; font-size:1.3rem; color:var(--ink); display:block; margin-bottom: 0.2rem;}

/* ---------- Probability Bars ---------- */
.prob{margin:1.2rem 0;}
.prob .row{display:flex; justify-content:space-between; font-weight:600; margin-bottom:.5rem; font-family: 'Outfit';}
.prob .track{height:16px; background:#E2E8F0; border-radius:99px; overflow:hidden;}
.prob .fill{height:100%; border-radius:99px;}
.fill.b{background:var(--benign);} .fill.m{background:var(--malignant);}

.panel{
  background:var(--glass); border:1px solid rgba(255,255,255,0.8);
  border-radius:20px; padding:2rem;
  box-shadow: 0 10px 30px -10px rgba(0,0,0,0.05);
}
.panel h3{margin-top:0; margin-bottom: 1.5rem;}

/* ---------- Legend ---------- */
.legend{height:12px; border-radius:99px; margin-top: 1rem;
  background:linear-gradient(90deg,#2a1fa0,#2ec4d6,#9be24f,#ffd23f,#e8332b);}
.legend-labels{display:flex; justify-content:space-between; font-size:.85rem; color:#64748B; margin-top:.5rem; font-weight: 500;}

/* ---------- Sidebar & Socials ---------- */
section[data-testid="stSidebar"]{background:#0F172A;}
section[data-testid="stSidebar"] *{color:#CBD5E1 !important;}
section[data-testid="stSidebar"] h2{font-family:'Outfit',sans-serif !important; color:#fff !important;}
.side-stat{font-family:'Outfit',sans-serif; font-size:3.5rem; font-weight:800; color:var(--eosin) !important; line-height:1;}
.side-note{font-size:.95rem; opacity:.9; line-height:1.6;}
.side-note a { color: #FDF2F8 !important; text-decoration: none; font-weight: 600; transition: color 0.2s; background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 6px; display: inline-block; margin-top: 5px;}
.side-note a:hover { color: #fff !important; background: var(--eosin); }

.disclaimer{
  margin-top:2rem; padding:1.2rem; border-left:4px solid var(--eosin);
  background:rgba(255,255,255,0.6); border-radius:0 12px 12px 0; font-size:.95rem; color: #475569;
}

/* ---------- Footer ---------- */
.custom-footer {
    text-align: center; margin-top: 5rem; padding: 2rem 0;
    border-top: 1px solid rgba(0,0,0,0.05); font-size: 0.95rem; color: #64748B;
}
.custom-footer a { color: var(--hema); text-decoration: none; font-weight: 600; margin: 0 10px; transition: color 0.2s;}
.custom-footer a:hover { color: var(--eosin); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. Model
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

def field_of_view(img, title, caption):
    return f"""
    <div class="fov-wrap">
      <div class="fov"><img src="{to_data_uri(img)}" alt="{title}"></div>
      <div class="fov-cap"><b>{title}</b>{caption}</div>
    </div>"""

# ------------------------------------------------------------------
# 4. Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Model Metrics")
    st.markdown('<div class="side-stat">97.22%</div>'
                '<div class="side-note">Accuracy on held-out test split</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="side-note">'
        "<b>Architecture</b><br>MobileNetV2 (Transfer Learning)<br><br>"
        "<b>Dataset</b><br>BreaKHis Histopathology<br><br>"
        "<b>Explainability</b><br>Grad-CAM Heatmaps"
        "</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        '<div class="side-note">'
        "<b>Connect with the Developer</b><br>"
        '<a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank">💼 LinkedIn</a><br>'
        '<a href="https://www.instagram.com/kashif_rashtiic/" target="_blank">📸 Instagram</a>'
        "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 5. Hero Section (With Base64 SVG Ribbon for guaranteed loading)
# ------------------------------------------------------------------
# This is a clean, beautiful SVG path for a breast cancer ribbon
svg_ribbon = """<svg class="ribbon-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M50 10 C30 10 20 30 35 50 L15 90 L30 95 L50 65 L70 95 L85 90 L65 50 C80 30 70 10 50 10 Z" fill="#E0568F"/>
  <path d="M50 10 C30 10 20 30 35 50 L15 90 L30 95 L50 65 L70 95 L85 90 L65 50 C80 30 70 10 50 10 Z" fill="url(#grad)" opacity="0.4"/>
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ffffff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#E0568F;stop-opacity:0" />
    </linearGradient>
  </defs>
</svg>"""

st.markdown(f"""
<div class="hero-card">
  {svg_ribbon}
  <div class="hero-text">
    <h1>See what the model sees.</h1>
    <p>Upload a stained tissue slide. The AI classifies it as benign or malignant, then uses Grad-CAM to visually highlight which regions of the tissue drove that decision. Built for Breast Cancer Awareness Month.</p>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop a histopathology image here (JPG or PNG)", type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------
# 6. Inference + Results
# ------------------------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Analyzing tissue slide..."):
        pil_img = Image.open(uploaded_file).convert("RGB")
        rgb_img = (np.array(pil_img.resize((224, 224))) / 255.0).astype(np.float32)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = probs[pred_idx].item()

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred_idx)])[0]

    verdict = classes[pred_idx]
    css_class = verdict.lower()
    
    st.markdown(f"""
    <div class="verdict {css_class}">
      <div><div class="label">Model Prediction</div><div class="call">{verdict}</div></div>
      <div class="conf"><div class="label">Confidence</div><b>{confidence:.1%}</b></div>
    </div>
    """, unsafe_allow_html=True)

    strength = st.slider("Heatmap intensity", 0.2, 0.9, 0.5, 0.05,
                         help="Adjust how much of the heatmap overlays the original image.")
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=1 - strength)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(field_of_view(pil_img.resize((448, 448)), "Original Slide",
                                  "The image exactly as uploaded."),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(field_of_view(Image.fromarray(overlay).resize((448, 448)),
                                  "Grad-CAM Attention Map",
                                  "Warm colours indicate high model attention."),
                    unsafe_allow_html=True)
        st.markdown('<div class="legend"></div>'
                    '<div class="legend-labels"><span>Less influence</span>'
                    '<span>More influence</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pb, pm = probs[0].item(), probs[1].item()
    st.markdown(f"""
    <div class="panel">
      <h3>Probability Breakdown</h3>
      <div class="prob"><div class="row"><span>Benign</span><span>{pb:.1%}</span></div>
        <div class="track"><div class="fill b" style="width:{pb*100:.1f}%"></div></div></div>
      <div class="prob"><div class="row"><span>Malignant</span><span>{pm:.1%}</span></div>
        <div class="track"><div class="fill m" style="width:{pm*100:.1f}%"></div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer"><b>Medical Disclaimer:</b> This tool is a research project and
    not a diagnostic device. Do not use it for medical decisions. A qualified pathologist
    should review any real clinical case.</div>
    """, unsafe_allow_html=True)

else:
    # Empty State: Show 3 elegant steps when no image is uploaded
    st.markdown("""
    <div class="steps-container">
        <div class="step-card">
            <div class="step-num">1</div>
            <h4>Upload Image</h4>
            <p>Drop a JPG or PNG of an H&E stained tissue slide.</p>
        </div>
        <div class="step-card">
            <div class="step-num">2</div>
            <h4>AI Inference</h4>
            <p>MobileNetV2 analyzes the cellular structures in seconds.</p>
        </div>
        <div class="step-card">
            <div class="step-num">3</div>
            <h4>Explainability</h4>
            <p>Grad-CAM highlights the exact regions the AI focused on.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 7. Custom Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="custom-footer">
    <p>Built with PyTorch & Streamlit for Breast Cancer Awareness Month 🎀</p>
    <p>Developed by <b>Kashif Sultan</b></p>
    <p>
        <a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank">LinkedIn</a> • 
        <a href="https://www.instagram.com/kashif_rashtiic/" target="_blank">Instagram</a>
    </p>
</div>
""", unsafe_allow_html=True)
