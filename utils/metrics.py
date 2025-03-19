import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def calculate_psnr(y_true, y_pred, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        y_true (numpy.ndarray): Ground truth image.
        y_pred (numpy.ndarray): Predicted image.
        max_val (float): Maximum value of the signal.
        
    Returns:
        float: PSNR value in dB.
    """
    # Convert to the same data type and range
    if y_true.dtype != y_pred.dtype:
        if np.max(y_true) > 1.0 and np.max(y_pred) <= 1.0:
            y_pred = y_pred * 255.0
        elif np.max(y_true) <= 1.0 and np.max(y_pred) > 1.0:
            y_true = y_true * 255.0
            max_val = 255.0
    
    # Calculate PSNR using scikit-image
    return peak_signal_noise_ratio(y_true, y_pred, data_range=max_val)

def calculate_ssim(y_true, y_pred, max_val=1.0, multichannel=True):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        y_true (numpy.ndarray): Ground truth image.
        y_pred (numpy.ndarray): Predicted image.
        max_val (float): Maximum value of the signal.
        multichannel (bool): Whether the images are multichannel.
        
    Returns:
        float: SSIM value.
    """
    # Convert to the same data type and range
    if y_true.dtype != y_pred.dtype:
        if np.max(y_true) > 1.0 and np.max(y_pred) <= 1.0:
            y_pred = y_pred * 255.0
        elif np.max(y_true) <= 1.0 and np.max(y_pred) > 1.0:
            y_true = y_true * 255.0
            max_val = 255.0
    
    # Calculate SSIM using scikit-image
    return structural_similarity(
        y_true, y_pred, 
        data_range=max_val,
        channel_axis=2 if multichannel and len(y_true.shape) > 2 else None
    )

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) between two images.
    
    Args:
        y_true (numpy.ndarray): Ground truth image.
        y_pred (numpy.ndarray): Predicted image.
        
    Returns:
        float: MSE value.
    """
    # Convert to the same data type and range
    if y_true.dtype != y_pred.dtype:
        if np.max(y_true) > 1.0 and np.max(y_pred) <= 1.0:
            y_pred = y_pred * 255.0
        elif np.max(y_true) <= 1.0 and np.max(y_pred) > 1.0:
            y_true = y_true * 255.0
    
    return np.mean((y_true - y_pred) ** 2)

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) between two images.
    
    Args:
        y_true (numpy.ndarray): Ground truth image.
        y_pred (numpy.ndarray): Predicted image.
        
    Returns:
        float: MAE value.
    """
    # Convert to the same data type and range
    if y_true.dtype != y_pred.dtype:
        if np.max(y_true) > 1.0 and np.max(y_pred) <= 1.0:
            y_pred = y_pred * 255.0
        elif np.max(y_true) <= 1.0 and np.max(y_pred) > 1.0:
            y_true = y_true * 255.0
    
    return np.mean(np.abs(y_true - y_pred))

def evaluate_model(model, test_data, metrics=None):
    """
    Evaluate a model using the specified metrics.
    
    Args:
        model: TensorFlow model.
        test_data: Test data generator.
        metrics (list): List of metrics to calculate. Options: 'psnr', 'ssim', 'mse', 'mae'.
        
    Returns:
        dict: Dictionary of metrics.
    """
    if metrics is None:
        metrics = ['psnr', 'ssim', 'mse', 'mae']
    
    results = {metric: [] for metric in metrics}
    
    for lr_imgs, hr_imgs in test_data:
        # Predict
        sr_imgs = model.predict(lr_imgs)
        
        # Calculate metrics for each image in the batch
        for i in range(len(hr_imgs)):
            if 'psnr' in metrics:
                results['psnr'].append(
                    tf.image.psnr(hr_imgs[i], sr_imgs[i], max_val=1.0).numpy()
                )
            
            if 'ssim' in metrics:
                results['ssim'].append(
                    tf.image.ssim(hr_imgs[i], sr_imgs[i], max_val=1.0).numpy()
                )
            
            if 'mse' in metrics:
                results['mse'].append(
                    tf.reduce_mean(tf.square(hr_imgs[i] - sr_imgs[i])).numpy()
                )
            
            if 'mae' in metrics:
                results['mae'].append(
                    tf.reduce_mean(tf.abs(hr_imgs[i] - sr_imgs[i])).numpy()
                )
    
    # Calculate average metrics
    for metric in metrics:
        results[metric] = np.mean(results[metric])
    
    return results

def compare_metrics(images_dict, reference_image):
    """
    Compare metrics between reference image and a dictionary of images.
    
    Args:
        images_dict (dict): Dictionary of images to compare. {name: image}
        reference_image (numpy.ndarray): Reference image to compare against.
        
    Returns:
        dict: Dictionary of metrics for each image.
    """
    metrics = {}
    
    for name, image in images_dict.items():
        metrics[name] = {
            'psnr': calculate_psnr(reference_image, image),
            'ssim': calculate_ssim(reference_image, image),
            'mse': calculate_mse(reference_image, image),
            'mae': calculate_mae(reference_image, image)
        }
    
    return metrics 