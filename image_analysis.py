"""
CURASCAN - Image Analysis Module
Classification and segmentation algorithms with modality-specific optimization
"""

import numpy as np
import cv2
from PIL import Image
from config import *
# Ensure VALIDATION_CONFIG is explicitly recognized if star import is failing in linting
try:
    from config import VALIDATION_CONFIG
except ImportError:
    pass


def run_classification(image, modality='X-Ray'):
    """
    Run classification optimized for specific imaging modality
    
    Args:
        image: PIL Image object
        modality: String - 'X-Ray', 'CT', or 'MRI'
        
    Returns:
        tuple: (result, confidence, None)
            - result: "Anomaly Detected" or "Normal"
            - confidence: float between 0 and 1
            - None: placeholder for compatibility (heatmap generated from segmentation)
    """
    # Get modality-specific parameters
    params = get_modality_params(modality)
    
    # Convert image to grayscale for analysis
    image_np = np.array(image.convert('L'))
    
    # Calculate image statistics
    mean_intensity = np.mean(image_np)
    std_intensity = np.std(image_np)
    
    # Calculate histogram to analyze intensity distribution
    hist = cv2.calcHist([image_np], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Entropy - irregularity in intensity distribution
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Texture analysis using Laplacian variance (STRONGEST DISCRIMINATOR)
    laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    
    # Texture variance using Sobel
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    texture_variance = np.var(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # Edge irregularity
    edges = cv2.Canny(image_np, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Spatial irregularity
    h, w = image_np.shape
    quad_means = [
        np.mean(image_np[:h//2, :w//2]),
        np.mean(image_np[:h//2, w//2:]),
        np.mean(image_np[h//2:, :w//2]),
        np.mean(image_np[h//2:, w//2:])
    ]
    spatial_variance = np.var(quad_means) / (mean_intensity + 1e-5)
    
    # Local intensity variations (infiltrates)
    kernel_size = 15
    local_mean = cv2.blur(image_np.astype(np.float32), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((image_np.astype(np.float32))**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.abs(local_sq_mean - local_mean**2))
    high_std_regions = np.sum(local_std > np.percentile(local_std, 85)) / local_std.size
    
    # Calculate confidence score - MODALITY-SPECIFIC CALIBRATION
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
    
    # Ensure confidence is between 0 and 1
    confidence = np.clip(confidence, 0.0, 1.0)
    
    # Determine result based on modality-specific threshold
    result = "Anomaly Detected" if confidence > params['classification_threshold'] else "Normal"
    
    return result, confidence, None


def run_segmentation_raw(image, modality='X-Ray'):
    """
    Run segmentation to identify regions of interest with modality-specific processing
    
    Args:
        image: PIL Image object
        modality: String - 'X-Ray', 'CT', or 'MRI'
        
    Returns:
        numpy.ndarray: Grayscale segmentation mask (224x224)
    """
    # Get modality-specific parameters
    params = get_modality_params(modality)
    
    # Convert to grayscale
    image_np = np.array(image.convert('L'))
    image_resized = cv2.resize(image_np, IMAGE_SIZE)
    
    # Apply CLAHE for better contrast (modality-specific clip limit)
    clahe = cv2.createCLAHE(clipLimit=params['clahe_clip_limit'], tileGridSize=(8, 8))
    enhanced = clahe.apply(image_resized)
    
    # Multiple thresholding approaches
    _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    edges = cv2.Canny(enhanced, 40, 120)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
    
    _, bright_regions = cv2.threshold(enhanced, np.percentile(enhanced, 85), 255, cv2.THRESH_BINARY)
    _, dark_regions = cv2.threshold(enhanced, np.percentile(enhanced, 15), 255, cv2.THRESH_BINARY_INV)
    
    # Combine all detection methods
    combined = np.zeros_like(image_resized, dtype=np.float32)
    combined += otsu_thresh.astype(np.float32) * 0.15
    combined += adaptive_thresh.astype(np.float32) * 0.25
    combined += edges_dilated.astype(np.float32) * 0.30
    combined += bright_regions.astype(np.float32) * 0.15
    combined += dark_regions.astype(np.float32) * 0.15
    
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    return mask


def run_segmentation(image, modality='X-Ray'):
    """
    Run segmentation and return colored overlay
    
    Args:
        image: PIL Image object
        modality: String - 'X-Ray', 'CT', or 'MRI'
        
    Returns:
        numpy.ndarray: Colored segmentation mask (original size, BGR)
    """
    mask = run_segmentation_raw(image, modality)
    
    if mask is not None:
        mask_resized = cv2.resize(mask, (image.width, image.height))
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_VIRIDIS)
        return mask_colored
    
    return None


def generate_gradcam_from_segmentation(segmentation_mask):
    """
    Generate Grad-CAM heatmap from segmentation mask
    
    Args:
        segmentation_mask: numpy.ndarray (grayscale mask)
        
    Returns:
        tuple: (heatmap_colored, confidence)
            - heatmap_colored: BGR heatmap image
            - confidence: float between 0 and 1
    """
    try:
        if segmentation_mask is None:
            return None, 0.5
        
        # Convert to grayscale if needed
        if len(segmentation_mask.shape) == 3:
            gray_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray_mask = segmentation_mask
        
        # Normalize
        normalized_mask = gray_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred_mask = cv2.GaussianBlur(normalized_mask, (15, 15), 0)
        blurred_mask = (blurred_mask - blurred_mask.min()) / (blurred_mask.max() - blurred_mask.min() + 1e-8)
        
        # Create heatmap
        heatmap = (blurred_mask * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Calculate confidence
        high_intensity = np.sum(normalized_mask > 0.6)
        very_high_intensity = np.sum(normalized_mask > 0.8)
        total_pixels = normalized_mask.size
        
        # Connected components analysis
        binary_mask = (gray_mask > 128).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
        
        max_component_size = 0
        if num_labels > 1:
            max_component_size = np.max(stats[1:, cv2.CC_STAT_AREA])
        
        component_ratio = max_component_size / total_pixels
        intensity_variance = np.var(normalized_mask)
        mean_high_intensity = np.mean(normalized_mask[normalized_mask > 0.5]) if np.any(normalized_mask > 0.5) else 0
        
        # Calculate confidence
        confidence = 0.0
        confidence += min((high_intensity / total_pixels) * 2.5, 0.30)
        confidence += min((very_high_intensity / total_pixels) * 5.0, 0.25)
        confidence += min(component_ratio * 2.0, 0.25)
        confidence += min(intensity_variance * 1.5, 0.15)
        confidence += min(mean_high_intensity * 0.08, 0.05)
        
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return heatmap_colored, confidence
        
    except Exception as e:
        print(f"Grad-CAM generation error: {str(e)}")
        return None, 0.5


def create_overlay(original_image, overlay):
    """
    Create overlay of heatmap on original image
    
    Args:
        original_image: PIL Image object
        overlay: numpy.ndarray (BGR overlay)
        
    Returns:
        PIL.Image: Combined overlay image
    """
    original_np = np.array(original_image.resize(IMAGE_SIZE))
    
    if overlay is not None:
        overlay_resized = cv2.resize(overlay, IMAGE_SIZE)
        combined = cv2.addWeighted(original_np, OVERLAY_ALPHA, overlay_resized, OVERLAY_BETA, 0)
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
