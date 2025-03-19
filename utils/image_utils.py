import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path (str): Path to image file.
        
    Returns:
        numpy.ndarray: Loaded image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return img

def save_image(image, save_path):
    """
    Save an image to file.
    
    Args:
        image (numpy.ndarray): Image to save.
        save_path (str): Path to save image.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to uint8 if necessary
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)

def normalize_image(image):
    """
    Normalize image to range [0, 1].
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Normalized image.
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def denormalize_image(image):
    """
    Denormalize image from [0, 1] to [0, 255].
    
    Args:
        image (numpy.ndarray): Normalized image.
        
    Returns:
        numpy.ndarray: Denormalized image.
    """
    if np.max(image) <= 1.0:
        return (image * 255.0).astype(np.uint8)
    return image.astype(np.uint8)

def create_lr_image(hr_image, scale_factor, method='bicubic'):
    """
    Create a low-resolution image by downsampling.
    
    Args:
        hr_image (numpy.ndarray): High-resolution image.
        scale_factor (int): Scale factor for downsampling.
        method (str): Downsampling method. Options: 'bicubic', 'bilinear', 'nearest'.
        
    Returns:
        numpy.ndarray: Low-resolution image.
    """
    # Get interpolation method
    interpolation_methods = {
        'bicubic': cv2.INTER_CUBIC,
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST
    }
    
    if method not in interpolation_methods:
        raise ValueError(f"Method {method} not supported. Choose from: {list(interpolation_methods.keys())}")
    
    h, w = hr_image.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # Downscale
    lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=interpolation_methods[method])
    
    return lr_image

def add_noise(image, noise_type='gaussian', noise_param=0.01):
    """
    Add noise to an image.
    
    Args:
        image (numpy.ndarray): Input image.
        noise_type (str): Type of noise. Options: 'gaussian', 'salt_pepper'.
        noise_param (float): Parameter controlling noise intensity.
        
    Returns:
        numpy.ndarray: Noisy image.
    """
    # Clone image
    noisy_image = image.copy()
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        if np.max(image) <= 1.0:
            noise = np.random.normal(0, noise_param, image.shape)
            noisy_image = np.clip(image + noise, 0.0, 1.0)
        else:
            noise = np.random.normal(0, noise_param * 255, image.shape)
            noisy_image = np.clip(image + noise, 0, 255).astype(image.dtype)
    
    elif noise_type == 'salt_pepper':
        # Add salt and pepper noise
        if np.max(image) <= 1.0:
            salt = np.random.random(image.shape) < noise_param / 2
            pepper = np.random.random(image.shape) < noise_param / 2
            noisy_image[salt] = 1.0
            noisy_image[pepper] = 0.0
        else:
            salt = np.random.random(image.shape[:2]) < noise_param / 2
            pepper = np.random.random(image.shape[:2]) < noise_param / 2
            
            # Apply to each channel if image is color
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    noisy_image[salt, i] = 255
                    noisy_image[pepper, i] = 0
            else:
                noisy_image[salt] = 255
                noisy_image[pepper] = 0
    
    else:
        raise ValueError(f"Noise type {noise_type} not supported. Choose from: ['gaussian', 'salt_pepper']")
    
    return noisy_image

def plot_comparison(images, titles, figsize=(15, 5)):
    """
    Plot comparison of multiple images.
    
    Args:
        images (list): List of images to compare.
        titles (list): List of titles for each image.
        figsize (tuple): Figure size.
    """
    assert len(images) == len(titles), "Number of images and titles must match."
    
    plt.figure(figsize=figsize)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        
        # Denormalize if necessary
        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def crop_center(img, crop_height, crop_width):
    """
    Crop the center of an image.
    
    Args:
        img (numpy.ndarray): Input image.
        crop_height (int): Height of crop.
        crop_width (int): Width of crop.
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    height, width = img.shape[:2]
    startx = width // 2 - crop_width // 2
    starty = height // 2 - crop_height // 2
    return img[starty:starty + crop_height, startx:startx + crop_width]

def random_crop(img, crop_height, crop_width):
    """
    Randomly crop an image.
    
    Args:
        img (numpy.ndarray): Input image.
        crop_height (int): Height of crop.
        crop_width (int): Width of crop.
        
    Returns:
        numpy.ndarray: Cropped image.
    """
    height, width = img.shape[:2]
    
    if height < crop_height or width < crop_width:
        raise ValueError("Crop size should be smaller than image dimensions")
    
    startx = np.random.randint(0, width - crop_width)
    starty = np.random.randint(0, height - crop_height)
    
    return img[starty:starty + crop_height, startx:startx + crop_width]

def random_flip(img):
    """
    Randomly flip an image horizontally or vertically.
    
    Args:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Flipped image.
    """
    # Horizontal flip
    if np.random.random() < 0.5:
        img = np.fliplr(img)
    
    # Vertical flip
    if np.random.random() < 0.5:
        img = np.flipud(img)
    
    return img

def random_rotate(img, angles=[0, 90, 180, 270]):
    """
    Randomly rotate an image by one of the specified angles.
    
    Args:
        img (numpy.ndarray): Input image.
        angles (list): List of angles to choose from.
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    angle = np.random.choice(angles)
    
    if angle == 0:
        return img
    elif angle == 90:
        return np.rot90(img)
    elif angle == 180:
        return np.rot90(img, k=2)
    elif angle == 270:
        return np.rot90(img, k=3)
    
    return img 