import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

from models.srgan import SRGAN
from models.cnn_model import SRCNN, EDSR
from models.traditional import upscale_image
from utils.data_loader import create_tf_dataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_mse, calculate_mae
from utils.image_utils import load_image, normalize_image, create_lr_image, save_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Super-Resolution Models')
    
    parser.add_argument('--model_type', type=str, default='edsr', 
                        choices=['srgan', 'srcnn', 'edsr', 'traditional'],
                        help='Model type to evaluate')
    parser.add_argument('--model_path', type=str, required=False,
                        help='Path to the model file (not needed for traditional methods)')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Directory containing test images')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Scale factor for super-resolution')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--traditional_method', type=str, default='bicubic',
                        choices=['bicubic', 'bilinear', 'lanczos', 'nearest'],
                        help='Method for traditional upscaling')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save output images from evaluation')
    
    return parser.parse_args()

def evaluate_traditional(args):
    """Evaluate traditional super-resolution methods."""
    print(f"Evaluating {args.traditional_method} upscaling...")
    
    # List all image files in the test directory
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        image_paths.extend(list(os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.endswith(ext)))
    
    if not image_paths:
        print(f"No images found in {args.test_dir}")
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.results_dir, args.traditional_method), exist_ok=True)
    
    # Initialize metrics lists
    psnr_values = []
    ssim_values = []
    mse_values = []
    mae_values = []
    times = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Load HR image
        hr_img = load_image(img_path)
        
        # Create LR image
        lr_img = create_lr_image(hr_img, args.scale_factor)
        
        # Measure upscaling time
        start_time = time.time()
        
        # Apply traditional upscaling
        sr_img = upscale_image(lr_img, args.scale_factor, method=args.traditional_method)
        
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        
        # Calculate metrics
        psnr = calculate_psnr(hr_img, sr_img)
        ssim = calculate_ssim(hr_img, sr_img)
        mse = calculate_mse(hr_img, sr_img)
        mae = calculate_mae(hr_img, sr_img)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)
        mae_values.append(mae)
        
        # Save output image if requested
        if args.save_images:
            img_name = os.path.basename(img_path)
            save_path = os.path.join(args.results_dir, args.traditional_method, img_name)
            save_image(sr_img, save_path)
            
            # Also save a comparison image
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(lr_img)
            plt.title('Low Resolution')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sr_img)
            plt.title(f'SR ({args.traditional_method})\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(hr_img)
            plt.title('High Resolution (Ground Truth)')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, args.traditional_method, f'comparison_{img_name}'))
            plt.close()
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    avg_time = np.mean(times)
    
    # Print results
    print(f"\nEvaluation Results for {args.traditional_method}:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average Processing Time: {avg_time:.4f} seconds per image")
    
    # Save results to CSV
    results = {
        'Method': args.traditional_method,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'MSE': avg_mse,
        'MAE': avg_mae,
        'Time': avg_time
    }
    
    df = pd.DataFrame([results])
    csv_path = os.path.join(args.results_dir, f'{args.traditional_method}_results.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")

def evaluate_dl_model(args):
    """Evaluate deep learning based super-resolution models."""
    print(f"Evaluating {args.model_type} model from {args.model_path}...")
    
    # Load model
    if args.model_type == 'srgan':
        # For SRGAN we only need the generator
        model = SRGAN(upscale_factor=args.scale_factor)
        model.generator = tf.keras.models.load_model(args.model_path)
    elif args.model_type == 'srcnn':
        model = SRCNN(upscale_factor=args.scale_factor)
        model.model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={'psnr': model.psnr}
        )
    elif args.model_type == 'edsr':
        model = EDSR(upscale_factor=args.scale_factor)
        model.model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={'psnr': model.psnr, 'ssim': model.ssim}
        )
    
    # Create test dataset
    test_dataset = create_tf_dataset(
        args.test_dir, 
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.results_dir, args.model_type), exist_ok=True)
    
    # Initialize metrics lists
    psnr_values = []
    ssim_values = []
    mse_values = []
    mae_values = []
    times = []
    
    # Process test dataset
    for i, (lr_batch, hr_batch) in enumerate(tqdm(test_dataset, desc="Processing batches")):
        # Predict SR images
        start_time = time.time()
        
        if args.model_type == 'srgan':
            sr_batch = model.generator.predict(lr_batch)
        else:
            sr_batch = model.model.predict(lr_batch)
        
        elapsed_time = time.time() - start_time
        
        # Calculate metrics for each image in the batch
        for j in range(len(lr_batch)):
            lr_img = lr_batch[j].numpy()
            hr_img = hr_batch[j].numpy()
            sr_img = sr_batch[j]
            
            # Calculate metrics
            psnr = tf.image.psnr(hr_img, sr_img, max_val=1.0).numpy()
            ssim = tf.image.ssim(hr_img, sr_img, max_val=1.0).numpy()
            mse = tf.reduce_mean(tf.square(hr_img - sr_img)).numpy()
            mae = tf.reduce_mean(tf.abs(hr_img - sr_img)).numpy()
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            mse_values.append(mse)
            mae_values.append(mae)
            times.append(elapsed_time / len(lr_batch))  # Time per image
            
            # Save output images if requested
            if args.save_images:
                # Convert to uint8 for saving
                lr_img_uint8 = (lr_img * 255).astype(np.uint8)
                hr_img_uint8 = (hr_img * 255).astype(np.uint8)
                sr_img_uint8 = (np.clip(sr_img, 0, 1) * 255).astype(np.uint8)
                
                # Save SR image
                save_path = os.path.join(args.results_dir, args.model_type, f'output_{i}_{j}.png')
                save_image(sr_img_uint8, save_path)
                
                # Save comparison image
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(lr_img)
                plt.title('Low Resolution')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(np.clip(sr_img, 0, 1))
                plt.title(f'SR ({args.model_type})\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(hr_img)
                plt.title('High Resolution (Ground Truth)')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.results_dir, args.model_type, f'comparison_{i}_{j}.png'))
                plt.close()
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    avg_time = np.mean(times)
    
    # Print results
    print(f"\nEvaluation Results for {args.model_type}:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average Processing Time: {avg_time:.4f} seconds per image")
    
    # Save results to CSV
    results = {
        'Method': args.model_type,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'MSE': avg_mse,
        'MAE': avg_mae,
        'Time': avg_time
    }
    
    df = pd.DataFrame([results])
    csv_path = os.path.join(args.results_dir, f'{args.model_type}_results.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to {csv_path}")

def main():
    args = parse_args()
    
    if args.model_type == 'traditional':
        evaluate_traditional(args)
    else:
        if not args.model_path:
            print("Error: --model_path is required for deep learning models")
            return
        evaluate_dl_model(args)

if __name__ == '__main__':
    main() 