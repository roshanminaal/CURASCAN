"""
CURASCAN - Configuration Module
Central configuration for the medical imaging platform with multi-modal support
"""

import os

# Application Settings
APP_TITLE = "CURASCAN - INTELLIGENT HEALTH CARE IMAGING Platform "
APP_ICON = "🏥"
LAYOUT = "wide"

# Database Configuration
DB_PATH = "curascan.db"
SCANS_DIR = "scans"

# Model Configuration
CHECKPOINT_DIR = "checkpoints"
CLS_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "cls_best.pth")
SEG_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "seg_best.pth")

# Image Processing Settings
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Modality-Specific Configuration
MODALITY_CONFIG = {
    'X-Ray': {
        'name': 'Chest X-Ray',
        'icon': '🔬',
        'description': 'Radiography imaging for pneumonia detection',
        'classification_threshold': 0.50,
        'laplacian_high': 300,
        'laplacian_med': 200,
        'laplacian_low': 100,
        'texture_high': 3500,
        'texture_med': 2500,
        'texture_low': 1500,
        'entropy_high': 7.0,
        'entropy_med': 7.2,
        'spatial_high': 0.5,
        'spatial_med': 0.35,
        'clahe_clip_limit': 2.0,
        'processing_notes': 'Optimized for lung field analysis and infiltrate detection'
    },
    'CT': {
        'name': 'CT Scan',
        'icon': '🏥',
        'description': 'Computed Tomography for cross-sectional analysis',
        'classification_threshold': 0.48,
        'laplacian_high': 85,
        'laplacian_med': 65,
        'laplacian_low': 45,
        'texture_high': 950,
        'texture_med': 750,
        'texture_low': 650,
        'entropy_high': 7.50,
        'entropy_med': 7.40,
        'spatial_high': 0.55,
        'spatial_med': 0.40,
        'clahe_clip_limit': 2.5,
        'processing_notes': 'Hounsfield unit-aware processing for lesion detection'
    },
    'MRI': {
        'name': 'MRI Scan',
        'icon': '🧲',
        'description': 'Magnetic Resonance Imaging for soft tissue analysis',
        'classification_threshold': 0.52,
        'laplacian_high': 65,
        'laplacian_med': 45,
        'laplacian_low': 30,
        'texture_high': 800,
        'texture_med': 650,
        'texture_low': 550,
        'entropy_high': 7.40,
        'entropy_med': 7.30,
        'spatial_high': 0.45,
        'spatial_med': 0.30,
        'clahe_clip_limit': 1.8,
        'processing_notes': 'Sequence-adaptive processing for T1/T2 weighted images'
    }
}

# Default modality (for backward compatibility)
DEFAULT_MODALITY = 'X-Ray'

# Legacy settings (use X-Ray parameters by default)
CLASSIFICATION_THRESHOLD = MODALITY_CONFIG['X-Ray']['classification_threshold']
LAPLACIAN_HIGH = MODALITY_CONFIG['X-Ray']['laplacian_high']
LAPLACIAN_MED = MODALITY_CONFIG['X-Ray']['laplacian_med']
LAPLACIAN_LOW = MODALITY_CONFIG['X-Ray']['laplacian_low']
TEXTURE_HIGH = MODALITY_CONFIG['X-Ray']['texture_high']
TEXTURE_MED = MODALITY_CONFIG['X-Ray']['texture_med']
TEXTURE_LOW = MODALITY_CONFIG['X-Ray']['texture_low']
ENTROPY_HIGH = MODALITY_CONFIG['X-Ray']['entropy_high']
ENTROPY_MED = MODALITY_CONFIG['X-Ray']['entropy_med']
SPATIAL_HIGH = MODALITY_CONFIG['X-Ray']['spatial_high']
SPATIAL_MED = MODALITY_CONFIG['X-Ray']['spatial_med']

# Confidence Weights (universal across modalities)
WEIGHT_LAPLACIAN = 0.35   # Primary discriminator
WEIGHT_TEXTURE = 0.25     # Secondary
WEIGHT_ENTROPY = 0.15     # Tertiary
WEIGHT_SPATIAL = 0.10
WEIGHT_LOCAL_STD = 0.10
WEIGHT_EDGE = 0.05

# UI Settings
OVERLAY_ALPHA = 0.2      # Original image weight in overlay
OVERLAY_BETA = 0.8       # Overlay weight

# Report Settings
REPORT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Ensure directories exist
os.makedirs(SCANS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_modality_params(modality):
    """Get parameters for specific imaging modality
    
    Args:
        modality: String - 'X-Ray', 'CT', or 'MRI'
        
    Returns:
        dict: Configuration parameters for the specified modality
    """
    return MODALITY_CONFIG.get(modality, MODALITY_CONFIG[DEFAULT_MODALITY])
