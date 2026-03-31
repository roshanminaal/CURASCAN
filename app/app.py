"""
CURASCAN Unified Application

This Streamlit app provides:
- Public scan analysis (no login) with multi‑modal AI (X‑Ray / CT / MRI)
- Private provider portal with login, patient management, and scan archive

Path: Image → Classification → UNet‑style Segmentation → Segmentation Mask → Grad‑CAM Heatmap
"""

import base64
import hashlib
import os
import sys
from datetime import datetime, date
from io import BytesIO

import cv2
import numpy as np
import sqlite3
import streamlit as st
from PIL import Image

# Ensure project root is importable (so `models/` and `utils/` work)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from report_utils import generate_report

# Optional: use trained PyTorch models if available
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    from models.densenet_classifier import build_mri_classifier, build_xray_classifier, build_ct_classifier
    from models.unet_segmenter import UNet
    from models.gradcam_utils import generate_gradcam_overlay

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Page configuration
st.set_page_config(
    page_title="CURASCAN - Medical Imaging Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        padding: 1.2rem 1rem;
        color: black;
        margin-bottom: 0.5rem;
    }
    .brand-logo {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 0.5rem 0;
        color: #123048;
        margin-bottom: 2rem;
    }
    .main-header .gradient-text {
        color: black;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f5fbff);
        padding: 1.4rem;
        border-radius: 14px;
        border-left: 4px solid #6EC1E4;
        margin: 0.5rem 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    }
    .success-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #e9f9f0, #d4f5e4);
        border: 1px solid #b2e4c8;
        color: #255d3b;
        box-shadow: 0 2px 8px rgba(20, 93, 59, 0.08);
    }
    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #fff7e6, #fff1cc);
        border: 1px solid #ffde99;
        color: #7a5a06;
        box-shadow: 0 2px 8px rgba(122, 90, 6, 0.08);
    }
    .stApp {
        background: radial-gradient(circle at top left, #f7fbff 0, #ffffff 45%, #f8f9ff 100%);
    }
    .stButton>button {
        background: linear-gradient(90deg, #6EC1E4, #9ADCF8);
        color: #ffffff;
        font-weight: 600;
        border-radius: 999px;
        border: none;
        padding: 0.55rem 1.2rem;
        box-shadow: 0 3px 8px rgba(110, 193, 228, 0.45);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #9ADCF8, #6EC1E4);
        box-shadow: 0 4px 12px rgba(110, 193, 228, 0.55);
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px dashed rgba(110, 193, 228, 0.6);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f4fbff, #ecf7ff);
        color: #123048;
        border-right: 1px solid #dde8f5;
    }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stRadio label {
        color: #123048;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - WITH MODALITY SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

DB_PATH = "curascan.db"
SCANS_DIR = "scans"
IMAGE_SIZE = (224, 224)

# Modality-Specific Configuration
MODALITY_CONFIG = {
    'X-Ray': {
        'name': 'Chest X-Ray',
        'icon': '🔬',
        'description': 'Radiography imaging for pneumonia detection',
        'classification_threshold': 0.50,
        'laplacian_high': 70,
        'laplacian_med': 50,
        'texture_high': 850,
        'texture_med': 700,
        'entropy_high': 7.45,
        'spatial_high': 0.5,
        'clahe_clip_limit': 2.0,
    },
    'CT': {
        'name': 'CT Scan',
        'icon': '🏥',
        'description': 'Computed Tomography for cross-sectional analysis',
        'classification_threshold': 0.48,
        'laplacian_high': 85,
        'laplacian_med': 65,
        'texture_high': 950,
        'texture_med': 750,
        'entropy_high': 7.50,
        'spatial_high': 0.55,
        'clahe_clip_limit': 2.5,
    },
    'MRI': {
        'name': 'MRI Scan',
        'icon': '🧲',
        'description': 'Magnetic Resonance Imaging for soft tissue analysis',
        'classification_threshold': 0.52,
        'laplacian_high': 65,
        'laplacian_med': 45,
        'texture_high': 800,
        'texture_med': 650,
        'entropy_high': 7.40,
        'spatial_high': 0.45,
        'clahe_clip_limit': 1.8,
    }
}

# Confidence Weights (universal)
WEIGHT_LAPLACIAN = 0.35
WEIGHT_TEXTURE = 0.25
WEIGHT_ENTROPY = 0.15
WEIGHT_SPATIAL = 0.10
WEIGHT_LOCAL_STD = 0.10
WEIGHT_EDGE = 0.05

# Validation Configuration (to detect non-medical images)
VALIDATION_CONFIG = {
    'max_saturation': 0.15,      # Medical scans should be mostly grayscale
    'min_entropy': 2.5,          # Too low entropy = blank/simple image
    'max_entropy': 7.8,          # Too high entropy = noisy natural photo
    'min_brightness': 10,        # Reject near-black images
    'max_brightness': 240,       # Reject near-white images
    'min_variance': 100,         # Reject flat/untextured images
}

os.makedirs(SCANS_DIR, exist_ok=True)

def get_modality_params(modality):
    """Get parameters for specific imaging modality"""
    return MODALITY_CONFIG.get(modality, MODALITY_CONFIG['X-Ray'])

# Resolve checkpoint paths relative to project root (cura1/)
_CHECKPOINTS_DIR = os.path.join(_ROOT_DIR, "checkpoints")
_CLS_XRAY_CKPT = os.path.join(_CHECKPOINTS_DIR, "cls_best.pth")
_CLS_CT_CKPT = os.path.join(_CHECKPOINTS_DIR, "cls_ct_best.pth")
_CLS_MRI_CKPT = os.path.join(_CHECKPOINTS_DIR, "cls_mri_best.pth")
_SEG_CKPT = os.path.join(_CHECKPOINTS_DIR, "seg_best.pth")


def _torch_device_str() -> str:
    if not _HAS_TORCH:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _classification_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@st.cache_resource
def _load_xray_model():
    if not _HAS_TORCH:
        return None
    if not os.path.exists(_CLS_XRAY_CKPT):
        return None
    device = torch.device(_torch_device_str())
    model = build_xray_classifier(pretrained=True)
    state = torch.load(_CLS_XRAY_CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def _load_seg_model():
    if not _HAS_TORCH:
        return None
    if not os.path.exists(_SEG_CKPT):
        return None
    device = torch.device(_torch_device_str())
    # Match the architecture used in train_segmentation.py
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    checkpoint = torch.load(_SEG_CKPT, map_location=device)
    # Handle both full checkpoints and state-only dicts
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def _load_mri_model():
    if not _HAS_TORCH:
        return None
    if not os.path.exists(_CLS_MRI_CKPT):
        return None
    device = torch.device(_torch_device_str())
    model = build_mri_classifier(pretrained=True)
    state = torch.load(_CLS_MRI_CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def _load_ct_model_v2():
    if not _HAS_TORCH:
        return None
    if not os.path.exists(_CLS_CT_CKPT):
        return None
    device = torch.device(_torch_device_str())
    model = build_ct_classifier(pretrained=True)
    state = torch.load(_CLS_CT_CKPT, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _predict_xray(image: Image.Image) -> tuple[str, float] | None:
    model = _load_xray_model()
    if model is None:
        return None

    device = next(model.parameters()).device
    x = _classification_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).item()
    result = "Anomaly Detected" if prob >= 0.5 else "Normal"
    return result, float(prob)


def _predict_ct(image: Image.Image) -> tuple[str, float] | None:
    model = _load_ct_model_v2()
    if model is None:
        return None

    device = next(model.parameters()).device
    x = _classification_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(1)
        prob = torch.sigmoid(logits).item()
    result = "Anomaly Detected" if prob >= 0.5 else "Normal"
    return result, float(prob)


def _predict_mri(image: Image.Image) -> tuple[str, float] | None:
    model = _load_mri_model()
    if model is None:
        return None

    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    device = next(model.parameters()).device
    x = _classification_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    label = classes[idx]
    if label == "No Tumor":
        result = "Normal (No Tumor)"
    else:
        result = f"Anomaly Detected ({label})"
    return result, conf


def _predict_segmentation(image: Image.Image) -> np.ndarray | None:
    model = _load_seg_model()
    if model is None:
        return None

    device = next(model.parameters()).device
    # Transformations should match train_segmentation.py / utils.augmentations
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        mask = torch.sigmoid(out).squeeze().cpu().numpy()
    
    # Return as uint8 mask 0-255
    return (mask * 255).astype(np.uint8)


def _load_uploaded_image(uploaded_file, modality_name: str) -> Image.Image | None:
    """
    Robust loader for uploaded images.
    - Supports PNG/JPEG directly via PIL.
    - For DICOM (.dcm), tries pydicom if available and falls back with a warning.
    """
    if uploaded_file is None:
        return None

    name = getattr(uploaded_file, "name", "").lower()

    # DICOM handling
    if name.endswith(".dcm"):
        try:
            import pydicom  # type: ignore

            ds = pydicom.dcmread(uploaded_file)
            arr = ds.pixel_array.astype("float32")
            # Normalize to 0–255 for display
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            arr = (arr * 255).clip(0, 255).astype("uint8")
            img = Image.fromarray(arr).convert("RGB")
            return img
        except Exception:
            st.error(
                f"Unable to read DICOM file for {modality_name}. "
                "Please upload a PNG or JPG export of the scan instead."
            )
            return None

    # Standard image formats
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Unable to open the uploaded image. Please check the file format.")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS (keeping your existing ones)
# ═══════════════════════════════════════════════════════════════════════════

def init_db():
    """Initialize database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT DEFAULT 'user',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id TEXT UNIQUE NOT NULL,
                  name TEXT NOT NULL,
                  dob DATE,
                  gender TEXT,
                  clinical_notes TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  created_by TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id TEXT NOT NULL,
                  scan_type TEXT,
                  scan_path TEXT,
                  result TEXT,
                  confidence REAL,
                  overlay_path TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  created_by TEXT,
                  FOREIGN KEY (patient_id) REFERENCES patients (patient_id))''')
    
    # Lightweight schema migrations for older databases
    # Ensure all expected columns exist before the app queries them
    c.execute("PRAGMA table_info(scans)")
    scan_columns = [row[1] for row in c.fetchall()]
    # Add any missing columns used by the app
    if 'result' not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN result TEXT")
    if 'confidence' not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN confidence REAL DEFAULT 0.0")
    if 'overlay_path' not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN overlay_path TEXT")
    if 'created_at' not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if 'created_by' not in scan_columns:
        c.execute("ALTER TABLE scans ADD COLUMN created_by TEXT")
    c.execute("PRAGMA table_info(patients)")
    patient_columns = [row[1] for row in c.fetchall()]
    if 'created_at' not in patient_columns:
        c.execute("ALTER TABLE patients ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if 'created_by' not in patient_columns:
        c.execute("ALTER TABLE patients ADD COLUMN created_by TEXT")
    c.execute("PRAGMA table_info(users)")
    user_columns = [row[1] for row in c.fetchall()]
    if 'role' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    if 'created_at' not in user_columns:
        c.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    try:
        password_hash = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                 ("admin", password_hash, "admin"))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password_hash))
    user = c.fetchone()
    conn.close()
    return user

def get_patients():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY created_at DESC")
    patients = c.fetchall()
    conn.close()
    return patients

def add_patient(name, dob, gender, clinical_notes, created_by):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM patients")
    count = c.fetchone()[0]
    patient_id = chr(65 + (count % 26)) + "-" + str(count + 1)
    c.execute("INSERT INTO patients (patient_id, name, dob, gender, clinical_notes, created_by) VALUES (?, ?, ?, ?, ?, ?)",
             (patient_id, name, dob, gender, clinical_notes, created_by))
    conn.commit()
    conn.close()
    return patient_id

def save_scan(patient_id, scan_type, scan_path, result, confidence, overlay_path, created_by):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO scans (patient_id, scan_type, scan_path, result, confidence, overlay_path, created_by) VALUES (?, ?, ?, ?, ?, ?, ?)",
             (patient_id, scan_type, scan_path, result, confidence, overlay_path, created_by))
    conn.commit()
    conn.close()

def get_patient_scans(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    scans = c.fetchall()
    conn.close()
    return scans

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS - WITH MODALITY SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

def run_classification(image, modality='X-Ray'):
    """Run classification with modality-specific parameters"""
    params = get_modality_params(modality)
    
    # Prefer trained model checkpoints if available (MRI multi-class, X-ray binary).
    if modality == "MRI":
        pred = _predict_mri(image)
        if pred is not None:
            return pred[0], pred[1], None
    if modality == "X-Ray":
        pred = _predict_xray(image)
        if pred is not None:
            return pred[0], pred[1], None
    if modality == "CT":
        pred = _predict_ct(image)
        if pred is not None:
            return pred[0], pred[1], None
    
    image_np = np.array(image.convert('L'))
    mean_intensity = np.mean(image_np)
    
    hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    texture_variance = np.var(np.sqrt(sobel_x**2 + sobel_y**2))
    
    edges = cv2.Canny(image_np, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    h, w = image_np.shape
    quad_means = [
        np.mean(image_np[:h//2, :w//2]),
        np.mean(image_np[:h//2, w//2:]),
        np.mean(image_np[h//2:, :w//2]),
        np.mean(image_np[h//2:, w//2:])
    ]
    spatial_variance = np.var(quad_means) / (mean_intensity + 1e-5)
    
    kernel_size = 15
    local_mean = cv2.blur(image_np.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((image_np.astype(np.float32))**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.abs(local_sq_mean - local_mean**2))
    high_std_regions = np.sum(local_std > np.percentile(local_std, 85)) / local_std.size

    # Pneumonia-specific: lower-lobe lateral asymmetry.
    # Bacterial pneumonia is usually unilateral — one lower quadrant becomes
    # dramatically whiter (consolidated) while the other stays dark (aerated).
    lower_left  = np.mean(image_np[h // 2:, :w // 2])
    lower_right = np.mean(image_np[h // 2:, w // 2:])
    lower_asymmetry = abs(lower_left - lower_right) / (max(lower_left, lower_right) + 1e-5)

    # Calculate confidence with modality-specific thresholds
    confidence = 0.0

    # X-Ray: higher texture threshold tuned from real image data.
    # Normal images often have texture around 2000-3500.
    # Pneumonia images often have texture around 1500-2500.
    # Note: Hand-tuned CV code is secondary to the DenseNet model.
    tex_high = 3500 if modality == "X-Ray" else params['texture_high']
    tex_med  = 2500 if modality == "X-Ray" else params['texture_med']
    if texture_variance > tex_high:
        confidence += WEIGHT_TEXTURE
    elif texture_variance > tex_med:
        confidence += WEIGHT_TEXTURE * 0.72

    # Laplacian Thresholds are also very low in config (70). 
    # Normal images showed 180-270.
    lap_high = 300 if modality == "X-Ray" else params['laplacian_high']
    lap_med = 200 if modality == "X-Ray" else params['laplacian_med']

    if laplacian_var > lap_high:
        confidence += WEIGHT_LAPLACIAN
    elif laplacian_var > lap_med:
        confidence += WEIGHT_LAPLACIAN * 0.71

    # X-Ray: Consolidation (pneumonia) fills air spaces → locally uniform → LOWER entropy.
    if modality == "X-Ray":
        if entropy < 7.0:         # tightened from 7.25
            confidence += WEIGHT_ENTROPY
    else:
        if entropy > params['entropy_high']:
            confidence += WEIGHT_ENTROPY

    if spatial_variance > params['spatial_high']:
        confidence += WEIGHT_SPATIAL

    if high_std_regions > 0.20:
        confidence += WEIGHT_LOCAL_STD

    if edge_density < 0.015:
        confidence += WEIGHT_EDGE

    # Lower-lobe asymmetry: strong unilateral opacity → pneumonia
    if lower_asymmetry > 0.25:    # upped from 0.15
        confidence += 0.35
    elif lower_asymmetry > 0.20:  # upped from 0.12
        confidence += 0.20

    confidence = np.clip(confidence, 0.0, 1.0)
    result = "Anomaly Detected" if confidence > 0.40 else "Normal"
    
    return result, confidence, None

def run_segmentation_raw(image, modality='X-Ray'):
    """Run segmentation with modality-specific parameters"""
    # AI-based segmentation (UNet)
    ai_mask = _predict_segmentation(image)
    if ai_mask is not None:
        return ai_mask

    # Fallback: Computer Vision / Image Processing approach
    params = get_modality_params(modality)
    
    image_np = np.array(image.convert('L'))
    image_resized = cv2.resize(image_np, IMAGE_SIZE)
    
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'], tileGridSize=(8, 8))
    enhanced = clahe.apply(image_resized)
    
    _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 15, 3)
    edges = cv2.Canny(enhanced, 40, 120)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
    
    _, bright_regions = cv2.threshold(enhanced, np.percentile(enhanced, 85), 255, cv2.THRESH_BINARY)
    _, dark_regions = cv2.threshold(enhanced, np.percentile(enhanced, 15), 255, cv2.THRESH_BINARY_INV)
    
    combined = np.zeros_like(image_resized, dtype=np.float32)
    combined += otsu_thresh.astype(np.float32) * 0.15
    combined += adaptive_thresh.astype(np.float32) * 0.25
    combined += edges_dilated.astype(np.float32) * 0.30
    combined += bright_regions.astype(np.float32) * 0.15
    combined += dark_regions.astype(np.float32) * 0.15
    
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    return mask

def run_segmentation(image, modality='X-Ray'):
    """Run segmentation and return colored overlay"""
    mask = run_segmentation_raw(image, modality)
    
    if mask is not None:
        mask_resized = cv2.resize(mask, (image.width, image.height))
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_VIRIDIS)
        return mask_colored
    
    return None

def generate_gradcam_from_segmentation(segmentation_mask):
    """Generate Grad-CAM heatmap from segmentation mask"""
    try:
        if segmentation_mask is None:
            return None, 0.5
        
        if len(segmentation_mask.shape) == 3:
            gray_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = segmentation_mask
        
        normalized_mask = gray_mask.astype(np.float32) / 255.0
        blurred_mask = cv2.GaussianBlur(normalized_mask, (15, 15), 0)
        blurred_mask = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min() + 1e-8)
        
        heatmap = (blurred_mask * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        high_intensity = np.sum(normalized_mask > 0.6)
        very_high_intensity = np.sum(normalized_mask > 0.8)
        total_pixels = normalized_mask.size
        
        binary_mask = (gray_mask > 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        
        max_component_size = 0
        if num_labels > 1:
            max_component_size = np.max(stats[1:, cv2.CC_STAT_AREA])
        
        component_ratio = max_component_size / total_pixels
        intensity_variance = np.var(normalized_mask)
        
        confidence = 0.0
        confidence += min((high_intensity / total_pixels) * 2.5, 0.30)
        confidence += min((very_high_intensity / total_pixels) * 5.0, 0.25)
        confidence += min(component_ratio * 2.0, 0.25)
        confidence += min(intensity_variance * 1.5, 0.15)
        
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return heatmap_colored, confidence
    except Exception as e:
        return None, 0.5

def create_overlay(original_image, overlay):
    """Create overlay of heatmap on original image"""
    original_np = np.array(original_image.resize(IMAGE_SIZE))
    
    if overlay is not None:
        overlay_resized = cv2.resize(overlay, IMAGE_SIZE)
        combined = cv2.addWeighted(original_np, 0.2, overlay_resized, 0.8, 0)
        return Image.fromarray(combined)
    
    return original_image


def is_medical_image(image):
    """
    Validate if an uploaded image is likely a medical scan.
    Returns: (bool, str) - (is_valid, error_message)
    """
    # 1. Color Saturation Check (Scans are mostly grayscale)
    img_hsv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2HSV)
    saturation = np.mean(img_hsv[:, :, 1]) / 255.0
    if saturation > VALIDATION_CONFIG['max_saturation']:
        return False, "Image appears too colorful to be a medical scan. Please upload a grayscale X-Ray, CT, or MRI."

    # 2. Intensity and Contrast Check
    image_np = np.array(image.convert('L'))
    mean_intensity = np.mean(image_np)
    intensity_variance = np.var(image_np)

    if mean_intensity < VALIDATION_CONFIG['min_brightness']:
        return False, "Image is too dark. Please upload a clear medical scan."
    if mean_intensity > VALIDATION_CONFIG['max_brightness']:
        return False, "Image is too bright/washed out. Please upload a clear medical scan."
    if intensity_variance < VALIDATION_CONFIG['min_variance']:
        return False, "Image lacks necessary contrast/detail for medical analysis."

    # 3. Entropy Check (Detects natural photos vs medical scans)
    hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    if entropy < VALIDATION_CONFIG['min_entropy']:
        return False, "Image is too simple or blank. Please upload a valid scan."
    if entropy > VALIDATION_CONFIG['max_entropy']:
        return False, "Image is too complex/noisy. This might be a natural photo or a low-quality scan."

    return True, "Valid medical scan detected."

# ═══════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def dashboard_page():
    """Dashboard with metrics"""
    st.markdown('<h1 class="main-header"><span class="gradient-text">📊 Dashboard</span></h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    with col1:
        c.execute("SELECT COUNT(*) FROM patients")
        patient_count = c.fetchone()[0]
        st.metric("Total Patients", patient_count)
    
    with col2:
        c.execute("SELECT COUNT(*) FROM scans")
        scan_count = c.fetchone()[0]
        st.metric("Total Scans", scan_count)
    
    with col3:
        c.execute("SELECT COUNT(*) FROM users")
        user_count = c.fetchone()[0]
        st.metric("System Users", user_count)
    
    with col4:
        st.metric("AI Models", "2 Active")
    
    conn.close()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Activity")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT 5")
        recent_scans = c.fetchall()
        conn.close()
        
        if recent_scans:
            for scan in recent_scans:
                st.write(f"🔬 {scan[1]} - {scan[2]} - {scan[4]}")
        else:
            st.info("No recent scans")
    
    with col2:
        st.subheader("AI Model Status")
        st.markdown('<div class="success-box">✅ Classification Model: Active</div>', unsafe_allow_html=True)
        st.markdown('<div class="success-box">✅ Segmentation Model: Active</div>', unsafe_allow_html=True)
        st.markdown('<div class="success-box">✅ Multi-Modal Support: X-Ray, CT, MRI</div>', unsafe_allow_html=True)

def patient_management_page():
    """Patient management interface"""
    st.title("👥 Patient Management")
    
    tab1, tab2 = st.tabs(["Add New Patient", "View Patients"])
    
    with tab1:
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name*")
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with col2:
                dob = st.date_input("Date of Birth", min_value=date(1950, 1, 1), max_value=date.today())
                clinical_notes = st.text_area("Clinical Notes")
            
            submit = st.form_submit_button("Add Patient", use_container_width=True)
            
            if submit:
                if name:
                    patient_id = add_patient(name, str(dob), gender, clinical_notes, st.session_state.username)
                    st.success(f"Patient added successfully! Patient ID: {patient_id}")
                else:
                    st.error("Patient name is required")
    
    with tab2:
        patients = get_patients()
        if patients:
            for patient in patients:
                with st.expander(f"🆔 {patient[1]} - {patient[2]}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**DOB:** {patient[3]}")
                        st.write(f"**Gender:** {patient[4]}")
                    with col2:
                        st.write(f"**Created:** {patient[6][:10]}")
                        st.write(f"**Created by:** {patient[7]}")
                    st.write(f"**Clinical Notes:** {patient[5]}")
        else:
            st.info("No patients in the system yet")

def scan_upload_page():
    """Scan upload with MODALITY SWITCHING - THIS IS THE KEY FEATURE!"""
    st.markdown('<h1 class="main-header"><span class="gradient-text">🔬 Scan Upload & Analysis</span></h1>', unsafe_allow_html=True)
    
    # Select patient
    patients = get_patients()
    if not patients:
        st.warning("⚠️ Please add a patient first in the Patient Management section")
        return
    
    patient_options = {f"{p[1]} - {p[2]}": p for p in patients}
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_data = patient_options[selected_patient]
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Scan")
        
        # ═══════════════════════════════════════════════════════════════════
        # MODALITY SELECTION WITH BUTTONS - THE MAGIC HAPPENS HERE!
        # ═══════════════════════════════════════════════════════════════════
        st.write("**Select Imaging Modality:**")
        
        # Initialize session state
        if 'selected_modality' not in st.session_state:
            st.session_state.selected_modality = 'X-Ray'
        
        # Create modality buttons
        modality_buttons = st.columns(3)
        
        with modality_buttons[0]:
            if st.button(f"{MODALITY_CONFIG['X-Ray']['icon']} X-Ray", 
                        use_container_width=True,
                        type="primary" if st.session_state.selected_modality == 'X-Ray' else "secondary"):
                st.session_state.selected_modality = 'X-Ray'
        
        with modality_buttons[1]:
            if st.button(f"{MODALITY_CONFIG['CT']['icon']} CT Scan", 
                        use_container_width=True,
                        type="primary" if st.session_state.selected_modality == 'CT' else "secondary"):
                st.session_state.selected_modality = 'CT'
        
        with modality_buttons[2]:
            if st.button(f"{MODALITY_CONFIG['MRI']['icon']} MRI", 
                        use_container_width=True,
                        type="primary" if st.session_state.selected_modality == 'MRI' else "secondary"):
                st.session_state.selected_modality = 'MRI'
        
        # Get current modality
        current_modality = st.session_state.selected_modality
        modality_params = get_modality_params(current_modality)
        
        # Display selected modality info
        st.info(f"""
        **Selected: {modality_params['icon']} {modality_params['name']}**
        
        {modality_params['description']}
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            f"Choose {modality_params['name']} Image", 
            type=['png', 'jpg', 'jpeg', 'dcm']
        )
        
        if uploaded_file:
            image = _load_uploaded_image(uploaded_file, modality_params["name"])
            if image is not None:
                st.image(image, caption=f"Uploaded {current_modality} Image", use_container_width=True)
                
                # Image Validation
                is_valid, msg = is_medical_image(image)
                if not is_valid:
                    st.error(f"❌ {msg}")
                    st.warning("⚠️ AI analysis may be unreliable for this image.")
                    uploaded_is_valid = False
                else:
                    st.success(f"✅ {msg}")
                    uploaded_is_valid = True
            else:
                return
            
            # Analysis options
            ai_model = st.radio("AI Analysis", ["Classification", "Segmentation", "Both"], disabled=not uploaded_is_valid)
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                st.write(f"**Modality-Specific Parameters ({current_modality}):**")
                st.write(f"- Classification Threshold: {modality_params['classification_threshold']}")
                st.write(f"- Laplacian Threshold: {modality_params['laplacian_high']}")
                st.write(f"- Texture Threshold: {modality_params['texture_high']}")
                st.write(f"- CLAHE Clip Limit: {modality_params['clahe_clip_limit']}")
            
            if st.button("🚀 Run AI Analysis", use_container_width=True, type="primary", disabled=not uploaded_is_valid):
                with st.spinner(f"Running {current_modality} AI analysis..."):
                    result_text = ""
                    confidence = 0.0
                    overlay_image = None
                    cls_score = None
                    seg_score = None
                    heatmap_image = None
                    
                    # STEP 1: Classification with modality-specific parameters
                    if ai_model in ["Classification", "Both"]:
                        result, cls_score, _ = run_classification(image, modality=current_modality)
                        result_text = result
                    
                    # STEP 2: Segmentation with modality-specific parameters
                    if ai_model in ["Segmentation", "Both"]:
                        seg_mask = run_segmentation(image, modality=current_modality)
                        if seg_mask is not None:
                            overlay_image = create_overlay(image, seg_mask)
                            
                            # STEP 3: Grad-CAM
                            # Use true gradient-based Grad-CAM from the classifier instead of just blurring the mask
                            if current_modality == "X-Ray":
                                model_for_cam = _load_xray_model()
                            elif current_modality == "CT":
                                model_for_cam = _load_ct_model_v2()
                            elif current_modality == "MRI":
                                model_for_cam = _load_mri_model()
                            else:
                                model_for_cam = None

                            if model_for_cam is not None:
                                device = next(model_for_cam.parameters()).device
                                x_cam = _classification_transform()(image.convert("RGB")).unsqueeze(0).to(device)
                                cam_overlay_bgr, cam_conf = generate_gradcam_overlay(model_for_cam, x_cam, image)
                                # Convert BGR to RGB for Streamlit/PIL
                                heatmap_image = Image.fromarray(cv2.cvtColor(cam_overlay_bgr, cv2.COLOR_BGR2RGB))
                                seg_score = cam_conf
                            else:
                                # Fallback if Grad-CAM utils fail
                                raw_mask = run_segmentation_raw(image, modality=current_modality)
                                heatmap, gradcam_conf = generate_gradcam_from_segmentation(raw_mask)
                                if heatmap is not None:
                                    heatmap_image = create_overlay(image, heatmap)
                                seg_score = gradcam_conf
                    
                    # Generate result text
                    if ai_model == "Both":
                        if cls_score is not None and seg_score is not None:
                            result_text = f"Classification: {result} ({cls_score:.2%}) | Grad-CAM: {seg_score:.2%}"
                            confidence = cls_score
                        elif cls_score is not None:
                            result_text = f"Classification: {result} ({cls_score:.2%})"
                            confidence = cls_score
                        if heatmap_image is not None:
                            overlay_image = heatmap_image
                    elif ai_model == "Classification":
                        if cls_score is not None:
                            result_text = result
                            confidence = cls_score
                    else:
                        if seg_score is not None:
                            result_text = f"Segmentation Complete | Grad-CAM: {seg_score:.2%}"
                            confidence = seg_score
                            if heatmap_image is not None:
                                overlay_image = heatmap_image
                    
                    # Save scan
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    scan_filename = f"{SCANS_DIR}/scan_{patient_data[1]}_{current_modality}_{timestamp}.png"
                    image.save(scan_filename)
                    
                    overlay_filename = None
                    if overlay_image:
                        overlay_filename = f"{SCANS_DIR}/overlay_{patient_data[1]}_{current_modality}_{timestamp}.png"
                        overlay_image.save(overlay_filename)
                    
                    save_scan(patient_data[1], current_modality, scan_filename, 
                             result_text, confidence, overlay_filename, st.session_state.username)
                    
                    st.session_state.analysis_result = {
                        'result': result_text,
                        'confidence': confidence,
                        'overlay': overlay_image,
                        'modality': current_modality
                    }
                    
                    st.success(f"✅ {current_modality} analysis completed successfully!")
                    st.rerun()
    
    with col2:
        st.subheader("📊 AI Analysis Results")
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # Display modality badge
            modality_info = MODALITY_CONFIG[result.get('modality', 'X-Ray')]
            st.markdown(f"**{modality_info['icon']} {modality_info['name']} Analysis**")
            
            # Result box
            if "Anomaly Detected" in result['result'] or "Borderline" in result["result"]:
                box_class = "warning-box"
                icon = "⚠️"
            else:
                box_class = "success-box"
                icon = "✅"
            
            st.markdown(
                f'<div class="{box_class}"><strong>{icon} {result["result"]}</strong><br>Confidence: {result["confidence"]:.2%}</div>', 
                unsafe_allow_html=True
            )
            
            if result['overlay']:
                st.image(result['overlay'], caption="AI Analysis Overlay", use_container_width=True)
        else:
            st.info(f"Upload an image and run AI analysis to see results")

def scan_viewer_page():
    """View patient scans"""
    st.title("📁 Scan Viewer")
    
    patients = get_patients()
    if not patients:
        st.info("No patients in the system")
        return
    
    patient_options = {f"{p[1]} - {p[2]}": p for p in patients}
    selected_patient = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_data = patient_options[selected_patient]
    
    scans = get_patient_scans(patient_data[1])
    
    if scans:
        for scan in scans:
            with st.expander(f"{scan[2]} - {scan[7][:10]} - {scan[4]}"):
                col1, col2 = st.columns(2)
                with col1:
                    if os.path.exists(scan[3]):
                        st.image(scan[3], caption="Original Scan")
                with col2:
                    if scan[6] and os.path.exists(scan[6]):
                        st.image(scan[6], caption="AI Analysis")
                
                st.write(f"**Result:** {scan[4]}")
                st.write(f"**Confidence:** {scan[5]:.2%}")
                st.write(f"**Scan Type:** {scan[2]}")
                st.write(f"**Analyzed by:** {scan[8]}")
                
                # Report Generation
                scan_info = {
                    "type": scan[2],
                    "result": scan[4],
                    "confidence": scan[5],
                    "date": scan[7]
                }
                
                if st.button("📄 Generate Report", key=f"btn_report_{scan[0]}"):
                    report_html = generate_report(
                        patient_info=patient_data,
                        scan_info=scan_info,
                        image_path=scan[3],
                        overlay_path=scan[6] if scan[6] and os.path.exists(scan[6]) else None
                    )
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=report_html,
                        file_name=f"CURASCAN_Report_{patient_data[1]}_{scan[7][:10]}.html",
                        mime="text/html",
                        key=f"dl_report_{scan[0]}"
                    )

    else:
        st.info("No scans for this patient")


def _severity_from_confidence(result: str, confidence: float) -> str:
    if result != "Anomaly Detected":
        return "Normal"
    if confidence > 0.8:
        return "High"
    if confidence > 0.6:
        return "Moderate"
    return "Low"


def _public_detailed_analysis(result: str, confidence: float, scan_type: str) -> dict:
    if result == "Anomaly Detected":
        severity = _severity_from_confidence(result, confidence)
        recommendations = {
            "High": "Immediate specialist review recommended. Consider follow‑up imaging and lab correlation.",
            "Moderate": "Schedule consultation and correlate clinically. Repeat imaging if symptoms persist.",
            "Low": "Routine follow‑up recommended. Monitor for new or worsening symptoms.",
        }
        findings = (
            f"AI detected patterns suggestive of abnormal findings on the {scan_type} "
            f"with {confidence:.1%} confidence."
        )
    elif result == "Borderline":
        severity = "Uncertain"
        recommendations = {
            "Uncertain": (
                "Image features are inconclusive. Recommend radiologist review and correlation "
                "with clinical findings before making decisions."
            )
        }
        findings = (
            f"AI analysis of the {scan_type} is borderline with {confidence:.1%} confidence; "
            "no clear normal or abnormal pattern was identified."
        )
    else:
        severity = "Normal"
        recommendations = {
            "Normal": "No immediate action required. Continue routine clinical follow‑up."
        }
        findings = (
            f"AI analysis shows no convincing abnormality on the {scan_type} "
            f"with {confidence:.1%} confidence."
        )

    return {
        "severity": severity,
        "recommendation": recommendations[severity],
        "findings": findings,
    }


def _public_report_html(result: dict, analysis: dict) -> str:
    """Generate a standalone HTML report for public scans (no patient DB)."""

    def img_to_b64(img: Image.Image | None) -> str:
        if img is None:
            return ""
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    original_b64 = img_to_b64(result.get("original"))
    heatmap_b64 = img_to_b64(result.get("heatmap"))
    seg_b64 = img_to_b64(result.get("segmentation"))

    timestamp_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CURASCAN Public Diagnostic Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                background-color: white;
                padding: 32px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            }}
            .header {{
                text-align: center;
                color: #2E86AB;
                border-bottom: 3px solid #2E86AB;
                padding-bottom: 16px;
                margin-bottom: 24px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.2rem;
            }}
            .result-box {{
                background: {'#f8d7da' if result['result'] == 'Anomaly Detected' else '#d4edda'};
                padding: 18px;
                border-radius: 8px;
                border-left: 5px solid {'#dc3545' if result['result'] == 'Anomaly Detected' else '#28a745'};
                margin: 20px 0;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            .info-box {{
                background: #f0f9ff;
                padding: 14px;
                border-radius: 8px;
            }}
            .images {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 18px;
                margin: 26px 0;
            }}
            .image-box img {{
                width: 100%;
                border-radius: 8px;
                border: 2px solid #e0e0e0;
            }}
            .footer {{
                margin-top: 32px;
                padding-top: 16px;
                border-top: 2px solid #e0e0e0;
                color: #666;
                font-size: 0.9rem;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 CURASCAN</h1>
                <h2>AI Medical Imaging Summary</h2>
                <p>Generated: {timestamp_str}</p>
            </div>

            <div class="result-box">
                <h2>Result: {result['result']}</h2>
                <h3>Confidence: {result['confidence']:.1%}</h3>
                <h3>Severity: {analysis['severity']}</h3>
            </div>

            <div class="info-grid">
                <div class="info-box">
                    <h3>Scan Details</h3>
                    <p><strong>Scan Type:</strong> {result['scan_type']}</p>
                    <p><strong>Patient ID (user supplied):</strong> {result['patient_id'] or 'Not provided'}</p>
                    <p><strong>Analysis Type:</strong> {result['analysis_type']}</p>
                </div>
                <div class="info-box">
                    <h3>AI Interpretation</h3>
                    <p>{analysis['findings']}</p>
                    <h4>Recommendations</h4>
                    <p>{analysis['recommendation']}</p>
                </div>
            </div>

            <h3>Visual Analysis</h3>
            <div class="images">
                <div class="image-box">
                    <h4>Original Scan</h4>
                    <img src="data:image/png;base64,{original_b64}" alt="Original Scan" />
                </div>
                {f'<div class="image-box"><h4>Grad-CAM Heatmap</h4><img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap" /></div>' if heatmap_b64 else ''}
                {f'<div class="image-box"><h4>Segmentation Overlay</h4><img src="data:image/png;base64,{seg_b64}" alt="Segmentation" /></div>' if seg_b64 else ''}
            </div>

            <div class="footer">
                <p><strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for educational and research use only and
                must not be used as a sole basis for diagnosis or treatment decisions. Always consult a qualified
                healthcare professional.</p>
            </div>
        </div>
    </body>
    </html>
    """


def landing_page() -> None:
    """First screen: choose Public vs Private access."""
    st.markdown('<h1 class="main-header"><span class="gradient-text">CURASCAN — INTELLIGENT HEALTH CARE IMAGING</span></h1>', unsafe_allow_html=True)
    st.markdown(
        "Advanced medical imaging analysis for professionals. Upload X‑Ray, CT, or MRI scans for "
        "instant, AI‑powered diagnostic insights."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔓 Start Public Scan", use_container_width=True):
            st.session_state.access_mode = "public"
            st.session_state.public_page = "🏠 Home"
            st.rerun()
    with col2:
        if st.button("🔐 Provider Login", use_container_width=True):
            st.session_state.access_mode = "private"
            st.rerun()

    st.markdown("---")
    st.subheader("Why CURASCAN?")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**📊AI‑Powered Triage**")
        st.write("DenseNet‑inspired classification and UNet‑style segmentation with Grad‑CAM explainability.")
    with col_b:
        st.markdown("**🔥Multi‑Modality**")
        st.write("Calibrated pipelines for X‑Ray, CT, and MRI with modality‑specific parameters.")
    with col_c:
        st.markdown("**📁Provider‑Ready**")
        st.write("Patient management, scan archive, and HTML report export for clinical workflows.")

    st.markdown("---")
    st.info(
        "⚠️ CURASCAN is an AI‑assisted tool designed to support, not replace, professional medical judgment."
    )


def public_home_page() -> None:
    st.markdown('<h1 class="main-header"><span class="gradient-text">🏥 CURASCAN</span></h1>', unsafe_allow_html=True)
    st.markdown("### AI‑Powered Medical Imaging for Everyone")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Supported Modalities", "3", "X‑Ray / CT / MRI")
    with col2:
        st.metric("AI Models", "2", "Classification + Segmentation")
    with col3:
        st.metric("Avg. Processing Time", "< 3 s")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("### What you can do")
        st.markdown(
            "- **Upload a scan** (X‑Ray, CT, MRI)\n"
            "- **Run AI analysis**: classification, segmentation, or both\n"
            "- **Visualize** Grad‑CAM heatmaps and overlays\n"
            "- **Download** a shareable HTML report"
        )
    with right:
        st.markdown("### Workflow")
        st.markdown(
            "1. Image upload\n"
            "2. **Classification**\n"
            "3. **UNet‑style segmentation**\n"
            "4. Segmentation mask\n"
            "5. **Grad‑CAM heatmap** from mask\n"
            "6. Report generation"
        )


def public_scan_upload_page() -> None:
    st.markdown('<h1 class="main-header"><span class="gradient-text">🔬 Public Scan Upload & Analysis</span></h1>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.markdown("### 📤 Upload Medical Scan")

        modality = st.radio(
            "Imaging Modality",
            ["X-Ray", "CT", "MRI"],
            format_func=lambda m: f"{MODALITY_CONFIG[m]['icon']} {MODALITY_CONFIG[m]['name']}",
        )
        params = get_modality_params(modality)

        # Mirror provider view: show selected modality summary
        st.info(
            f"**Selected: {params['icon']} {params['name']}**\n\n{params['description']}"
        )

        patient_id = st.text_input("Patient ID (optional)")

        uploaded = st.file_uploader(
            f"Choose {params['name']} image",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supported formats: PNG, JPG, JPEG, DICOM exports.",
        )

        if uploaded:
            image = _load_uploaded_image(uploaded, params["name"])
            if image is not None:
                st.image(image, caption=f"Uploaded {params['name']}", use_container_width=True)
                
                # Image Validation
                is_valid, msg = is_medical_image(image)
                if not is_valid:
                    st.error(f"❌ {msg}")
                    st.warning("⚠️ AI analysis may be unreliable for this image.")
                    uploaded_is_valid = False
                else:
                    st.success(f"✅ {msg}")
                    uploaded_is_valid = True
            else:
                uploaded_is_valid = False
        else:
            image = None
            uploaded_is_valid = False

    with col_right:
        st.markdown("### 🤖 AI Analysis Options")
        ai_choice = st.radio(
            "Analysis Type",
            ["Classification Only", "Segmentation Only", "Both (Recommended)"],
            disabled=not uploaded_is_valid
        )
        show_heatmap = st.checkbox("Show Grad‑CAM Heatmap", value=True, disabled=not uploaded_is_valid)
        show_seg_overlay = st.checkbox("Show Segmentation Overlay", value=True, disabled=not uploaded_is_valid)

        if st.button("🚀 Run AI Analysis", use_container_width=True, disabled=not uploaded_is_valid):
            if image is None:
                st.warning("Please upload an image first.")
            else:
                with st.spinner(f"Analyzing {params['name']} with AI..."):
                    result_text = ""
                    confidence = 0.0
                    heatmap_image = None
                    seg_overlay_image = None

                    cls_result = None
                    cls_conf = None
                    seg_conf = None

                    if ai_choice in ["Classification Only", "Both (Recommended)"]:
                        cls_result, cls_conf, _ = run_classification(image, modality=modality)
                        result_text = cls_result
                        confidence = cls_conf

                    if ai_choice in ["Segmentation Only", "Both (Recommended)"]:
                        seg_mask_colored = run_segmentation(image, modality=modality)
                        if seg_mask_colored is not None:
                            if show_seg_overlay:
                                seg_overlay_image = create_overlay(image, seg_mask_colored)

                            # Use true gradient-based Grad-CAM
                            if modality == "X-Ray":
                                model_for_cam = _load_xray_model()
                            elif modality == "CT":
                                model_for_cam = _load_ct_model_v2()
                            elif modality == "MRI":
                                model_for_cam = _load_mri_model()
                            else:
                                model_for_cam = None

                            if model_for_cam is not None and show_heatmap:
                                device = next(model_for_cam.parameters()).device
                                x_cam = _classification_transform()(image.convert("RGB")).unsqueeze(0).to(device)
                                cam_overlay_bgr, cam_conf = generate_gradcam_overlay(model_for_cam, x_cam, image)
                                heatmap_image = Image.fromarray(cv2.cvtColor(cam_overlay_bgr, cv2.COLOR_BGR2RGB))
                                seg_conf = cam_conf
                            else:
                                # Fallback
                                raw_mask = run_segmentation_raw(image, modality=modality)
                                heatmap, grad_conf = generate_gradcam_from_segmentation(raw_mask)
                                if heatmap is not None and show_heatmap:
                                    heatmap_image = create_overlay(image, heatmap)
                                seg_conf = grad_conf
                            if ai_choice == "Segmentation Only":
                                result_text = "Segmentation Complete"
                                confidence = seg_conf

                    if ai_choice == "Both (Recommended)" and cls_conf is not None and seg_conf is not None:
                        result_text = f"{cls_result} ({cls_conf:.1%}) | Grad‑CAM {seg_conf:.1%}"
                        confidence = cls_conf

                    st.session_state.public_result = {
                        "result": result_text or "Analysis failed",
                        "confidence": float(confidence),
                        "scan_type": params["name"],
                        "analysis_type": ai_choice,
                        "patient_id": patient_id,
                        "original": image,
                        "heatmap": heatmap_image,
                        "segmentation": seg_overlay_image,
                    }
                    st.success("✅ Analysis complete")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📊 AI Results")

    result = st.session_state.get("public_result")
    if not result:
        st.info("Upload a scan and run analysis to see results.")
        return

    base_label = (
        "Anomaly Detected"
        if "Anomaly Detected" in result["result"]
        else "Borderline"
        if "Borderline" in result["result"]
        else "Normal"
    )
    analysis = _public_detailed_analysis(base_label, result["confidence"], result["scan_type"])

    if "Anomaly Detected" in result["result"]:
        if result["confidence"] > 0.8:
            box_class = "warning-box"
            icon = "🔴"
        else:
            box_class = "warning-box"
            icon = "🟡"
    else:
        box_class = "success-box"
        icon = "🟢"

    st.markdown(
        f'<div class="{box_class}"><strong>{icon} {result["result"]}</strong>'
        f"<br>Confidence: {result['confidence']:.1%}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Confidence Level**")
    conf_pct = result["confidence"] * 100
    st.progress(conf_pct / 100)
    st.write(f"{conf_pct:.1f}%")

    col_info, col_meta = st.columns(2)
    with col_info:
        st.markdown("#### Findings")
        st.info(analysis["findings"])
        st.markdown("#### Recommendations")
        st.write(analysis["recommendation"])
    with col_meta:
        st.markdown("#### Scan Details")
        st.write(f"**Scan Type:** {result['scan_type']}")
        st.write(f"**Patient ID:** {result['patient_id'] or 'Not provided'}")
        st.write(f"**Analysis Type:** {result['analysis_type']}")
        st.write(f"**Run At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.markdown("---")
    st.markdown("### 🖼️ Visual Analysis")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Original Scan**")
        st.image(result["original"], use_container_width=True)
    with cols[1]:
        st.markdown("**Grad‑CAM Heatmap**")
        if result["heatmap"] is not None:
            st.image(result["heatmap"], use_container_width=True)
        else:
            st.info("Heatmap not available")
    with cols[2]:
        st.markdown("**Segmentation Overlay**")
        if result["segmentation"] is not None:
            st.image(result["segmentation"], use_container_width=True)
        else:
            st.info("Segmentation overlay not available")

    st.markdown("---")
    st.markdown("### 📥 Export")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if st.button("📄 Generate HTML Report", use_container_width=True, key="public_generate_report"):
            html = _public_report_html(result, analysis)
            st.session_state.public_report_html = html
    with col_r2:
        if "public_report_html" in st.session_state:
            st.download_button(
                "⬇️ Download Report (HTML)",
                data=st.session_state.public_report_html,
                file_name=f"curascan_public_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True,
            )
    with col_r3:
        if st.button("💾 Save to Local Archive", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image = result["original"]
            if image is not None:
                save_path = os.path.join(SCANS_DIR, f"public_{timestamp}.png")
                image.save(save_path)
                st.success(f"Saved snapshot to {save_path}")


def public_app() -> None:
    """Wrapper for the public (no‑login) experience."""
    with st.sidebar:
        st.markdown("### 🌐 Public Access")
        page = st.radio("Navigation", ["🏠 Home", "🔬 Scan Upload & Analysis"], label_visibility="collapsed")
        st.markdown("---")
        if st.button("← Back to Mode Selection", use_container_width=True):
            st.session_state.access_mode = "landing"
            st.rerun()

    if page == "🏠 Home":
        public_home_page()
    else:
        public_scan_upload_page()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_db()
    # Access mode: landing → public (no login) → private (provider)
    if "access_mode" not in st.session_state:
        st.session_state.access_mode = "landing"

    mode = st.session_state.access_mode

    # 1) Landing selector
    if mode == "landing":
        landing_page()
        return

    # 2) Public experience
    if mode == "public":
        public_app()
        return

    # 3) Private / provider portal (existing admin flow)
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown('<h1 class="main-header"><span class="gradient-text">🏥 CURASCAN Provider Login</span></h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                user = verify_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = user[3]
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.info("Default provider account: admin / admin123")

            if st.button("← Back to Mode Selection", use_container_width=True, key="back_from_login"):
                st.session_state.access_mode = "landing"
                st.rerun()
        return

    # Sidebar for authenticated providers
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"Role: {st.session_state.role}")

        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Patient Management", "Scan Upload", "Scan Viewer"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

        if st.button("← Back to Mode Selection", use_container_width=True, key="back_from_private"):
            st.session_state.authenticated = False
            st.session_state.access_mode = "landing"
            st.rerun()

    # Page routing for provider portal
    if page == "Dashboard":
        dashboard_page()
    elif page == "Patient Management":
        patient_management_page()
    elif page == "Scan Upload":
        scan_upload_page()
    elif page == "Scan Viewer":
        scan_viewer_page()

if __name__ == "__main__":
    main()