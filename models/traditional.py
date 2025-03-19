import cv2
import numpy as np
from PIL import Image

def bicubic_upscale(image, scale_factor):
    """
    Upscale an image using bicubic interpolation.
    
    Args:
        image (numpy.ndarray): Input image.
        scale_factor (int): Scale factor for upscaling.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Resize using OpenCV
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return resized

def bilinear_upscale(image, scale_factor):
    """
    Upscale an image using bilinear interpolation.
    
    Args:
        image (numpy.ndarray): Input image.
        scale_factor (int): Scale factor for upscaling.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Resize using OpenCV
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized

def lanczos_upscale(image, scale_factor):
    """
    Upscale an image using Lanczos interpolation.
    
    Args:
        image (numpy.ndarray): Input image.
        scale_factor (int): Scale factor for upscaling.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    # Convert to PIL image for Lanczos
    if len(image.shape) == 3:
        pil_img = Image.fromarray(image)
    else:
        pil_img = Image.fromarray(image).convert('L')
    
    # Get image dimensions
    w, h = pil_img.size
    
    # Calculate new dimensions
    new_w, new_h = w * scale_factor, h * scale_factor
    
    # Resize using Lanczos
    resized_pil = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Convert back to numpy array
    resized = np.array(resized_pil)
    
    return resized

def nearest_neighbor_upscale(image, scale_factor):
    """
    Upscale an image using nearest neighbor interpolation.
    
    Args:
        image (numpy.ndarray): Input image.
        scale_factor (int): Scale factor for upscaling.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Resize using OpenCV
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    return resized

def upscale_image(image, scale_factor, method='bicubic'):
    """
    Upscale an image using the specified method.
    
    Args:
        image (numpy.ndarray): Input image.
        scale_factor (int): Scale factor for upscaling.
        method (str): Upscaling method. Options: 'bicubic', 'bilinear', 'lanczos', 'nearest'.
        
    Returns:
        numpy.ndarray: Upscaled image.
    """
    methods = {
        'bicubic': bicubic_upscale,
        'bilinear': bilinear_upscale,
        'lanczos': lanczos_upscale,
        'nearest': nearest_neighbor_upscale
    }
    
    if method not in methods:
        raise ValueError(f"Method {method} not supported. Choose from: {list(methods.keys())}")
    
    return methods[method](image, scale_factor) 