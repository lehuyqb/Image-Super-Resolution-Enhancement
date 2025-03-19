import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from models.srgan import SRGAN
from models.cnn_model import SRCNN, EDSR
from models.traditional import upscale_image
from utils.image_utils import load_image, normalize_image, save_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Super-Resolution Inference')
    
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image for super-resolution')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the output image (default: auto-generated)')
    parser.add_argument('--model', type=str, default='bicubic',
                        choices=['bicubic', 'bilinear', 'lanczos', 'nearest', 'srgan', 'srcnn', 'edsr'],
                        help='Super-resolution method to use')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model file (required for deep learning models)')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Scale factor for super-resolution')
    parser.add_argument('--show_result', action='store_true',
                        help='Display the result after processing')
    parser.add_argument('--compare_with_original', action='store_true',
                        help='Compare with the original image if available')
    parser.add_argument('--original_image', type=str, default=None,
                        help='Path to original high-resolution image for comparison')
    
    return parser.parse_args()

def load_model(args):
    """Load the specified model."""
    if args.model in ['bicubic', 'bilinear', 'lanczos', 'nearest']:
        # Traditional methods don't need a model
        return None
    
    if not args.model_path:
        raise ValueError(f"Model path must be provided for {args.model} model")
    
    print(f"Loading {args.model} model from {args.model_path}...")
    
    if args.model == 'srgan':
        model = SRGAN(upscale_factor=args.scale_factor)
        model.generator = tf.keras.models.load_model(args.model_path)
    elif args.model == 'srcnn':
        model = SRCNN(upscale_factor=args.scale_factor)
        model.model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={'psnr': model.psnr}
        )
    elif args.model == 'edsr':
        model = EDSR(upscale_factor=args.scale_factor)
        model.model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={'psnr': model.psnr, 'ssim': model.ssim}
        )
    
    return model

def perform_super_resolution(input_image, model, args):
    """Perform super-resolution on the input image."""
    if args.model in ['bicubic', 'bilinear', 'lanczos', 'nearest']:
        # Traditional method
        print(f"Applying {args.model} upscaling with scale factor {args.scale_factor}...")
        start_time = time.time()
        sr_image = upscale_image(input_image, args.scale_factor, method=args.model)
        processing_time = time.time() - start_time
    else:
        # Deep learning method
        print(f"Applying {args.model} super-resolution with scale factor {args.scale_factor}...")
        
        # Normalize input for deep learning model
        lr_image_norm = normalize_image(input_image)
        lr_image_norm = np.expand_dims(lr_image_norm, axis=0)  # Add batch dimension
        
        # Generate super-resolution image
        start_time = time.time()
        if args.model == 'srgan':
            sr_image_norm = model.generator.predict(lr_image_norm)[0]
        else:
            sr_image_norm = model.model.predict(lr_image_norm)[0]
        processing_time = time.time() - start_time
        
        # Convert back to uint8
        sr_image = (np.clip(sr_image_norm, 0, 1) * 255).astype(np.uint8)
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    return sr_image, processing_time

def display_and_save_result(input_image, sr_image, args, processing_time, original_image=None):
    """Display and save the super-resolution result."""
    # Determine output filename if not provided
    if args.output_path is None:
        input_basename = os.path.basename(args.input_image)
        input_name, input_ext = os.path.splitext(input_basename)
        output_filename = f"{input_name}_{args.model}_x{args.scale_factor}{input_ext}"
        output_path = output_filename
    else:
        output_path = args.output_path
    
    # Save output image
    print(f"Saving result to {output_path}...")
    save_image(sr_image, output_path)
    
    # Display result if requested
    if args.show_result:
        if args.compare_with_original and original_image is not None:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(input_image)
            plt.title('Input (Low Resolution)')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(sr_image)
            plt.title(f'Super-Resolution ({args.model})\nTime: {processing_time:.2f}s')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(original_image)
            plt.title('Original (High Resolution)')
            plt.axis('off')
        else:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(input_image)
            plt.title('Input (Low Resolution)')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(sr_image)
            plt.title(f'Super-Resolution ({args.model})\nTime: {processing_time:.2f}s')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    args = parse_args()
    
    # Load the input image
    print(f"Loading input image: {args.input_image}")
    input_image = load_image(args.input_image)
    
    # Load the original high-resolution image if provided
    original_image = None
    if args.compare_with_original and args.original_image:
        print(f"Loading original image: {args.original_image}")
        original_image = load_image(args.original_image)
    
    # Load the model
    model = load_model(args)
    
    # Perform super-resolution
    sr_image, processing_time = perform_super_resolution(input_image, model, args)
    
    # Display and save the result
    display_and_save_result(input_image, sr_image, args, processing_time, original_image)
    
    print("Inference completed successfully")

if __name__ == '__main__':
    main() 