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
st.set_page_config(page_title="Slide Reader · Explainable Breast Cancer AI",
                   page_icon="🎀", layout="wide")

# ------------------------------------------------------------------
# 2. Design system
#    Palette comes from H&E staining, the stain pathologists actually use:
#    hematoxylin (blue-purple) for nuclei, eosin (pink) for tissue.
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root{
  --glass:#F7F4FA;        /* slide glass */
  --ink:#1E1633;          /* dark nuclei */
  --hema:#3B2A6B;         /* hematoxylin */
  --eosin:#E0568F;        /* eosin */
  --stroma:#CBBBDD;       /* faint tissue */
  --benign:#2F5D8C;
  --malignant:#B3205A;
}

html, body, [class*="css"], .stApp{
  font-family:'Instrument Sans',sans-serif;
  color:var(--ink);
}
.stApp{
  background:
    radial-gradient(circle at 12% 8%, #EADCF3 0, transparent 38%),
    radial-gradient(circle at 92% 30%, #F8DDE9 0, transparent 34%),
    var(--glass);
}
#MainMenu, footer, header{visibility:hidden;}
.block-container{padding-top:2.2rem; max-width:1200px;}

h1,h2,h3,.serif{font-family:'Fraunces',serif !important; color:var(--ink);}

/* ---------- hero ---------- */
.hero{padding:1.2rem 0 2rem;}
.hero h1{
  font-size:clamp(2.4rem,5.5vw,4.4rem);
  font-weight:800; line-height:1.02; letter-spacing:-0.02em;
  margin:0 0 .9rem; max-width:15ch;
}
.hero p{font-size:1.1rem; max-width:52ch; color:#4a3f66; line-height:1.55; margin:0;}
.hero .ribbon{
  display:inline-flex; align-items:center; gap:.5rem;
  background:var(--ink); color:#fff; border-radius:999px;
  padding:.35rem .9rem; font-size:.85rem; margin-bottom:1.2rem;
}
.hero .ribbon i{width:.6rem;height:.6rem;border-radius:50%;background:var(--eosin);display:inline-block;}

/* ---------- uploader ---------- */
[data-testid="stFileUploader"] section{
  background:rgba(255,255,255,.65);
  border:2px dashed var(--hema);
  border-radius:18px; padding:1.6rem;
}
[data-testid="stFileUploader"] section:hover{border-color:var(--eosin); background:#fff;}
[data-testid="stFileUploader"] button{
  background:var(--hema); color:#fff; border:none; border-radius:10px;
}

/* ---------- verdict ---------- */
.verdict{
  display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;
  border-radius:20px; padding:1.6rem 2rem; color:#fff; margin:1.2rem 0 1.8rem;
}
.verdict.malignant{background:linear-gradient(110deg,var(--malignant),#7d1442);}
.verdict.benign{background:linear-gradient(110deg,var(--benign),#1d3d63);}
.verdict .label{font-size:.95rem; opacity:.85; margin-bottom:.2rem;}
.verdict .call{font-family:'Fraunces',serif; font-size:clamp(2rem,4.5vw,3.4rem); font-weight:800; line-height:1;}
.verdict .conf{text-align:right;}
.verdict .conf b{font-family:'Fraunces',serif; font-size:clamp(2rem,4.5vw,3.4rem); font-weight:600; display:block; line-height:1;}

/* ---------- circular microscope field of view ---------- */
.fov-wrap{text-align:center;}
.fov{
  width:100%; aspect-ratio:1/1; border-radius:50%; overflow:hidden;
  border:10px solid var(--ink);
  box-shadow:0 0 0 3px var(--stroma), 0 18px 40px rgba(59,42,107,.28);
  background:#000;
}
.fov img{width:100%; height:100%; object-fit:cover; display:block;}
.fov-cap{margin-top:1rem; font-size:.95rem; color:#4a3f66;}
.fov-cap b{font-family:'Fraunces',serif; font-size:1.2rem; color:var(--ink); display:block;}

/* ---------- probability bars ---------- */
.prob{margin:.9rem 0;}
.prob .row{display:flex; justify-content:space-between; font-weight:600; margin-bottom:.35rem;}
.prob .track{height:16px; background:#E6DDF0; border-radius:99px; overflow:hidden;}
.prob .fill{height:100%; border-radius:99px;}
.fill.b{background:var(--benign);} .fill.m{background:var(--malignant);}

.panel{
  background:rgba(255,255,255,.72); border:1px solid var(--stroma);
  border-radius:20px; padding:1.5rem 1.8rem;
}
.panel h3{margin-top:0;}

/* ---------- heatmap legend ---------- */
.legend{height:12px; border-radius:99px;
  background:linear-gradient(90deg,#2a1fa0,#2ec4d6,#9be24f,#ffd23f,#e8332b);}
.legend-labels{display:flex; justify-content:space-between; font-size:.8rem; color:#4a3f66; margin-top:.3rem;}

/* ---------- sidebar & socials ---------- */
section[data-testid="stSidebar"]{background:var(--ink);}
section[data-testid="stSidebar"] *{color:#EDE6F7 !important;}
section[data-testid="stSidebar"] h2{font-family:'Fraunces',serif !important; color:#fff !important;}
.side-stat{font-family:'Fraunces',serif; font-size:3rem; font-weight:800; color:var(--eosin) !important; line-height:1;}
.side-note{font-size:.9rem; opacity:.85; line-height:1.5;}
.side-note a { color: var(--eosin) !important; text-decoration: none; font-weight: 600; transition: color 0.2s;}
.side-note a:hover { color: #fff !important; }

.disclaimer{
  margin-top:2rem; padding:1rem 1.3rem; border-left:5px solid var(--eosin);
  background:rgba(255,255,255,.7); border-radius:0 12px 12px 0; font-size:.9rem;
}

/* ---------- custom footer ---------- */
.custom-footer {
    text-align: center; margin-top: 4rem; padding-top: 2rem;
    border-top: 1px solid var(--stroma); font-size: 0.9rem; color: #4a3f66;
}
.custom-footer a { color: var(--eosin); text-decoration: none; font-weight: 600; margin: 0 10px; }
.custom-footer a:hover { text-decoration: underline; }

@media (prefers-reduced-motion:no-preference){
  .verdict{animation:rise .6s ease-out both;}
  @keyframes rise{from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;}}
}
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
    """Convert a PIL image or numpy array to a base64 data URI for HTML display."""
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
    st.markdown("## About the model")
    st.markdown('<div class="side-stat">97.22%</div>'
                '<div class="side-note">accuracy on the held-out test split</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="side-note">'
        "<b>Architecture</b><br>MobileNetV2, fine-tuned for two classes.<br><br>"
        "<b>Data</b><br>BreaKHis breast histopathology images.<br><br>"
        "<b>Explanation method</b><br>Grad-CAM highlights the tissue regions that pushed "
        "the model toward its answer."
        "</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    # Added Developer Social Links here
    st.markdown(
        '<div class="side-note">'
        "<b>Connect with the Developer</b><br><br>"
        '<a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank">LinkedIn ↗</a><br>'
        '<a href="https://www.instagram.com/kashif_rashtiic/" target="_blank">Instagram ↗</a>'
        "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 5. Hero
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="ribbon"><i></i> Breast Cancer Awareness Month</div>
  <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Breast_Cancer_Awareness_Ribbon.svg/200px-Breast_Cancer_Awareness_Ribbon.svg.png" width="60" alt="Pink Ribbon">
    <h1 style="margin: 0;">See what the model sees.</h1>
  </div>
  <p style="margin-top: 1rem;">Upload a stained tissue slide. The model classifies it as benign or malignant,
  then shows which regions of the tissue drove that decision.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop a histopathology image here (JPG or PNG)",
                                 type=["jpg", "jpeg", "png"])

# ------------------------------------------------------------------
# 6. Inference + results
# ------------------------------------------------------------------
if uploaded_file is not None:
    with st.spinner("Reading the slide..."):
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
      <div><div class="label">Model prediction</div><div class="call">{verdict}</div></div>
      <div class="conf"><div class="label">Confidence</div><b>{confidence:.1%}</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Heatmap strength control (re-blends the overlay instantly)
    strength = st.slider("Heatmap strength", 0.2, 0.9, 0.5, 0.05,
                         help="Low values show more of the original tissue; "
                              "high values show more of the heatmap.")
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,
                                image_weight=1 - strength)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(field_of_view(pil_img.resize((448, 448)), "Original slide",
                                  "The image exactly as you uploaded it."),
                    unsafe_allow_html=True)
    with col2:
        st.markdown(field_of_view(Image.fromarray(overlay).resize((448, 448)),
                                  "What the model focused on",
                                  "Warm colours mark regions that influenced the decision most."),
                    unsafe_allow_html=True)
        st.markdown('<div class="legend" style="margin-top:.8rem"></div>'
                    '<div class="legend-labels"><span>Less influence</span>'
                    '<span>More influence</span></div>', unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    pb, pm = probs[0].item(), probs[1].item()
    st.markdown(f"""
    <div class="panel">
      <h3>How the model split its confidence</h3>
      <div class="prob"><div class="row"><span>Benign</span><span>{pb:.1%}</span></div>
        <div class="track"><div class="fill b" style="width:{pb*100:.1f}%"></div></div></div>
      <div class="prob"><div class="row"><span>Malignant</span><span>{pm:.1%}</span></div>
        <div class="track"><div class="fill m" style="width:{pm*100:.1f}%"></div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer"><b>Educational demo.</b> This tool is a research project and
    not a diagnostic device. Do not use it for medical decisions. A qualified pathologist
    should review any real case.</div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 7. Custom Footer (Added Developer Info)
# ------------------------------------------------------------------
st.markdown("""
<div class="custom-footer">
    <p>Built with PyTorch & Streamlit for Pinktober 2024 🎀</p>
    <p>Developed by <b>Kashif Sultan</b></p>
    <p>
        <a href="https://www.linkedin.com/in/kashif-sultan11" target="_blank">LinkedIn</a> • 
        <a href="https://www.instagram.com/kashif_rashtiic/" target="_blank">Instagram</a>
    </p>
</div>
""", unsafe_allow_html=True)
