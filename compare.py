import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tabulate import tabulate

from models.srgan import SRGAN
from models.cnn_model import SRCNN, EDSR
from models.traditional import upscale_image
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_mae
from utils.image_utils import load_image, normalize_image, create_lr_image, save_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare Super-Resolution Methods')
    
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image for comparison')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Scale factor for super-resolution')
    parser.add_argument('--methods', type=str, nargs='+', default=['bicubic', 'bilinear', 'lanczos'],
                        help='Methods to compare (traditional methods and model names)')
    parser.add_argument('--srgan_model', type=str, default=None,
                        help='Path to SRGAN generator model')
    parser.add_argument('--srcnn_model', type=str, default=None,
                        help='Path to SRCNN model')
    parser.add_argument('--edsr_model', type=str, default=None,
                        help='Path to EDSR model')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    
    return parser.parse_args()

def load_models(args):
    """Load the required models based on the selected methods."""
    models = {}
    
    # Check which deep learning methods are requested
    if 'srgan' in args.methods and args.srgan_model:
        srgan = SRGAN(upscale_factor=args.scale_factor)
        srgan.generator = tf.keras.models.load_model(args.srgan_model)
        models['srgan'] = srgan
    
    if 'srcnn' in args.methods and args.srcnn_model:
        srcnn = SRCNN(upscale_factor=args.scale_factor)
        srcnn.model = tf.keras.models.load_model(
            args.srcnn_model,
            custom_objects={'psnr': srcnn.psnr}
        )
        models['srcnn'] = srcnn
    
    if 'edsr' in args.methods and args.edsr_model:
        edsr = EDSR(upscale_factor=args.scale_factor)
        edsr.model = tf.keras.models.load_model(
            args.edsr_model,
            custom_objects={'psnr': edsr.psnr, 'ssim': edsr.ssim}
        )
        models['edsr'] = edsr
    
    return models

def perform_comparison(args, models):
    """Perform comparison of different super-resolution methods."""
    print(f"Comparing super-resolution methods on image: {args.input_image}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the original high-resolution image
    hr_img = load_image(args.input_image)
    
    # Create low-resolution image by downscaling
    lr_img = create_lr_image(hr_img, args.scale_factor, method='bicubic')
    
    # Save the LR and HR images
    save_image(lr_img, os.path.join(args.output_dir, 'low_resolution.png'))
    save_image(hr_img, os.path.join(args.output_dir, 'high_resolution.png'))
    
    # Dictionary to store super-resolved images and metrics
    sr_images = {}
    metrics = {}
    
    # Process each method
    for method in args.methods:
        print(f"Processing method: {method}")
        
        start_time = time.time()
        
        if method in ['bicubic', 'bilinear', 'lanczos', 'nearest']:
            # Traditional method
            sr_img = upscale_image(lr_img, args.scale_factor, method=method)
        else:
            # Deep learning method
            if method not in models:
                print(f"Warning: Model for method '{method}' not provided. Skipping.")
                continue
            
            # Normalize input for deep learning model
            lr_img_norm = normalize_image(lr_img)
            lr_img_norm = np.expand_dims(lr_img_norm, axis=0)  # Add batch dimension
            
            # Generate super-resolution image
            if method == 'srgan':
                sr_img_norm = models[method].generator.predict(lr_img_norm)[0]
            else:
                sr_img_norm = models[method].model.predict(lr_img_norm)[0]
            
            # Convert back to uint8
            sr_img = (np.clip(sr_img_norm, 0, 1) * 255).astype(np.uint8)
        
        elapsed_time = time.time() - start_time
        
        # Save SR image
        save_image(sr_img, os.path.join(args.output_dir, f'{method}_output.png'))
        
        # Calculate metrics
        psnr = calculate_psnr(hr_img, sr_img)
        ssim = calculate_ssim(hr_img, sr_img)
        mse = calculate_mse(hr_img, sr_img)
        mae = calculate_mae(hr_img, sr_img)
        
        # Store image and metrics
        sr_images[method] = sr_img
        metrics[method] = {
            'PSNR': psnr,
            'SSIM': ssim,
            'MSE': mse,
            'MAE': mae,
            'Time': elapsed_time
        }
    
    # Create comparison visualization
    create_comparison_visualization(lr_img, hr_img, sr_images, metrics, args.output_dir)
    
    # Create metrics table
    create_metrics_table(metrics, args.output_dir)
    
    print(f"Comparison completed. Results saved to {args.output_dir}")

def create_comparison_visualization(lr_img, hr_img, sr_images, metrics, output_dir):
    """Create a visual comparison of all methods."""
    # Determine grid size
    n_methods = len(sr_images) + 2  # +2 for LR and HR
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 5*n_rows))
    
    # Add LR image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(lr_img)
    plt.title('Low Resolution')
    plt.axis('off')
    
    # Add HR image
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(hr_img)
    plt.title('High Resolution (Ground Truth)')
    plt.axis('off')
    
    # Add each SR image
    for i, (method, sr_img) in enumerate(sr_images.items(), start=3):
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(sr_img)
        plt.title(f'{method.upper()}\nPSNR: {metrics[method]["PSNR"]:.2f}, SSIM: {metrics[method]["SSIM"]:.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_visualization.png'), dpi=300)
    plt.close()

def create_metrics_table(metrics, output_dir):
    """Create a table of metrics for all methods."""
    # Convert metrics to DataFrame
    data = []
    for method, m in metrics.items():
        data.append({
            'Method': method.upper(),
            'PSNR (dB)': m['PSNR'],
            'SSIM': m['SSIM'],
            'MSE': m['MSE'],
            'MAE': m['MAE'],
            'Time (s)': m['Time']
        })
    
    df = pd.DataFrame(data)
    
    # Sort by PSNR (descending)
    df = df.sort_values('PSNR (dB)', ascending=False)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
    
    # Also save as a pretty table to text file
    table = tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f')
    with open(os.path.join(output_dir, 'metrics_comparison.txt'), 'w') as f:
        f.write(table)
    
    # Print metrics table
    print("\nMetrics Comparison:")
    print(table)

def main():
    args = parse_args()
    
    # Load models based on the selected methods
    models = load_models(args)
    
    # Perform comparison
    perform_comparison(args, models)

if __name__ == '__main__':
    main() 