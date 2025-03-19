import os
import argparse
import subprocess
import sys
import time
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quick Demo for Image Super-Resolution')
    
    parser.add_argument('--input_image', type=str, default=None,
                        help='Path to input image (optional, will download samples if not provided)')
    parser.add_argument('--scale_factor', type=int, default=4,
                        choices=[2, 3, 4, 8],
                        help='Scale factor for super-resolution')
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['bicubic', 'bilinear', 'lanczos'],
                        help='Methods to compare for super-resolution')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading sample test images')
    
    return parser.parse_args()

def download_sample_datasets():
    """Download a small test dataset for demonstration."""
    print("\nDownloading sample test dataset (Set5)...")
    try:
        subprocess.run([
            sys.executable,
            'download_dataset.py',
            '--dataset', 'set5',
            '--output_dir', 'data'
        ], check=True)
        print("Download completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error downloading sample dataset.")
        return False

def find_sample_image():
    """Find a suitable sample image for demonstration."""
    # Look in test directory first
    test_dirs = glob.glob('data/test/*')
    for test_dir in test_dirs:
        if os.path.isdir(test_dir):
            image_files = glob.glob(os.path.join(test_dir, '*.png')) + \
                        glob.glob(os.path.join(test_dir, '*.jpg'))
            if image_files:
                return image_files[0]
    
    # If no test images found, look for any image in the project
    all_images = glob.glob('**/*.png', recursive=True) + \
                glob.glob('**/*.jpg', recursive=True)
    if all_images:
        return all_images[0]
    
    return None

def create_low_resolution_version(image_path, scale_factor):
    """Create a low-resolution version of the input image."""
    try:
        # Import required modules
        from utils.image_utils import load_image, create_lr_image, save_image
        
        # Load the original image
        hr_img = load_image(image_path)
        
        # Create low-resolution version
        lr_img = create_lr_image(hr_img, scale_factor)
        
        # Save low-resolution image
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        lr_filename = f"{name}_lr_x{scale_factor}{ext}"
        lr_path = os.path.join(os.path.dirname(image_path), lr_filename)
        save_image(lr_img, lr_path)
        
        print(f"Created low-resolution version: {lr_path}")
        return lr_path
    except Exception as e:
        print(f"Error creating low-resolution image: {e}")
        return None

def run_comparison(input_image, methods, scale_factor):
    """Run comparison of different super-resolution methods."""
    print(f"\nRunning comparison of {', '.join(methods)} on {input_image}...")
    
    try:
        # Create results directory
        os.makedirs('results/demo', exist_ok=True)
        
        # Run comparison
        cmd = [
            sys.executable,
            'compare.py',
            '--input_image', input_image,
            '--scale_factor', str(scale_factor),
            '--methods'
        ] + methods + [
            '--output_dir', 'results/demo'
        ]
        
        subprocess.run(cmd, check=True)
        print("Comparison completed successfully.")
        return True
    except subprocess.CalledProcessError:
        print("Error running comparison.")
        return False

def run_inference(input_image, method, scale_factor):
    """Run inference using a specific method."""
    print(f"\nRunning inference with {method} on {input_image}...")
    
    try:
        # Create output directory
        os.makedirs('results/demo', exist_ok=True)
        
        # Generate output filename
        filename = os.path.basename(input_image)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_{method}_x{scale_factor}{ext}"
        output_path = os.path.join('results/demo', output_filename)
        
        # Run inference
        cmd = [
            sys.executable,
            'inference.py',
            '--input_image', input_image,
            '--model', method,
            '--scale_factor', str(scale_factor),
            '--output_path', output_path,
            '--show_result'
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Inference completed successfully. Result saved to {output_path}")
        return True
    except subprocess.CalledProcessError:
        print("Error running inference.")
        return False

def main():
    """Main function to run the quick demo."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("QUICK DEMO FOR IMAGE SUPER-RESOLUTION")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('results/demo', exist_ok=True)
    
    # Download sample dataset if needed
    if not args.skip_download and not args.input_image:
        download_sample_datasets()
    
    # Find or use input image
    input_image = args.input_image
    if not input_image:
        input_image = find_sample_image()
        if not input_image:
            print("No sample images found and no input image provided.")
            print("Please provide an input image with --input_image or run again without --skip_download.")
            return
    
    print(f"Using input image: {input_image}")
    
    # Create low-resolution version if it doesn't exist
    if "_lr_" not in input_image:
        lr_image = create_low_resolution_version(input_image, args.scale_factor)
        if lr_image:
            input_image = lr_image
    
    # Run comparison
    run_comparison(input_image, args.methods, args.scale_factor)
    
    # Run inference for each method
    for method in args.methods:
        run_inference(input_image, method, args.scale_factor)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to results/demo")
    print("\nTo train models and perform more advanced operations, see QUICKSTART.md")

if __name__ == '__main__':
    main() 