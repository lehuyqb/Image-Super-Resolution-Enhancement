import os
import argparse
import subprocess
import sys
import glob
import random
import shutil

def parse_args():
    """Parse command line arguments for setting up the environment."""
    parser = argparse.ArgumentParser(description='Setup environment for Image Super-Resolution & Enhancement')
    
    parser.add_argument('--download_div2k', action='store_true',
                        help='Download DIV2K dataset for training')
    parser.add_argument('--download_test_only', action='store_true',
                        help='Download only test datasets (Set5, Set14, Urban100)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['set5', 'set14'],
                        choices=['div2k', 'set5', 'set14', 'urban100', 'manga109', 'all'],
                        help='Datasets to download')
    parser.add_argument('--scale_factor', type=int, default=4,
                        choices=[2, 3, 4, 8],
                        help='Scale factor for DIV2K dataset')
    parser.add_argument('--create_sample_images', action='store_true',
                        help='Create sample low-resolution images for testing')
    parser.add_argument('--skip_dir_creation', action='store_true',
                        help='Skip creation of directory structure')
    
    return parser.parse_args()

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    print("\nCreating directory structure...")
    
    # Define directories to create
    directories = [
        'data/train/hr',
        'data/train/lr',
        'data/val/hr',
        'data/val/lr',
        'data/test',
        'data/samples',
        'saved_models',
        'results/comparison',
        'results/evaluation',
        'results/samples'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created {directory}")
    
    print("Directory structure created successfully.")

def download_datasets(args):
    """Download specified datasets."""
    if args.download_div2k or 'div2k' in args.datasets or 'all' in args.datasets:
        print("\nDownloading DIV2K dataset...")
        subprocess.run([
            sys.executable,
            'download_dataset.py',
            '--dataset', 'div2k',
            '--output_dir', 'data',
            '--scale_factor', str(args.scale_factor)
        ], check=True)
    
    test_datasets = []
    if args.download_test_only:
        test_datasets = ['set5', 'set14']
    elif 'all' in args.datasets:
        test_datasets = ['set5', 'set14', 'urban100', 'manga109']
    else:
        for dataset in args.datasets:
            if dataset != 'div2k':
                test_datasets.append(dataset)
    
    if test_datasets:
        print("\nDownloading test datasets...")
        for dataset in test_datasets:
            print(f"Downloading {dataset}...")
            subprocess.run([
                sys.executable,
                'download_dataset.py',
                '--dataset', dataset,
                '--output_dir', 'data'
            ], check=True)
    
    print("Dataset downloads completed.")

def create_sample_images():
    """Create sample low-resolution images for testing from existing images."""
    print("\nCreating sample images for testing...")
    
    try:
        # Import required modules
        from utils.image_utils import load_image, create_lr_image, save_image
        
        # Create samples directory
        os.makedirs('data/samples', exist_ok=True)
        
        # Look for existing images in test datasets
        test_images = []
        for test_dir in glob.glob('data/test/*'):
            if os.path.isdir(test_dir):
                test_images.extend(
                    glob.glob(os.path.join(test_dir, '*.png')) + 
                    glob.glob(os.path.join(test_dir, '*.jpg'))
                )
        
        # If no test images found, look for any images in the project
        if not test_images:
            test_images = glob.glob('**/*.png', recursive=True) + glob.glob('**/*.jpg', recursive=True)
            # Filter out images that might be results or other non-source images
            test_images = [img for img in test_images if 'results' not in img and 'samples' not in img]
        
        # Select a few images (up to 5) to create samples
        if test_images:
            selected_images = random.sample(test_images, min(5, len(test_images)))
            
            for img_path in selected_images:
                try:
                    # Load image
                    hr_img = load_image(img_path)
                    
                    # Create low-resolution versions with different scale factors
                    for scale in [2, 4]:
                        lr_img = create_lr_image(hr_img, scale)
                        
                        # Save images
                        filename = os.path.basename(img_path)
                        name, ext = os.path.splitext(filename)
                        
                        # Save HR image
                        hr_filename = f"{name}_hr{ext}"
                        hr_path = os.path.join('data/samples', hr_filename)
                        save_image(hr_img, hr_path)
                        
                        # Save LR image
                        lr_filename = f"{name}_lr_x{scale}{ext}"
                        lr_path = os.path.join('data/samples', lr_filename)
                        save_image(lr_img, lr_path)
                        
                        print(f"  Created sample pair: {hr_filename} and {lr_filename}")
                
                except Exception as e:
                    print(f"  Error processing {img_path}: {e}")
                    continue
            
            print("Sample images created successfully.")
        else:
            print("No images found to create samples. Download datasets first.")
    
    except ImportError:
        print("  Error: Could not import image utility functions. Skipping sample creation.")
    except Exception as e:
        print(f"  Error creating sample images: {e}")

def verify_requirements():
    """Verify that requirements.txt exists and contains necessary packages."""
    print("\nVerifying requirements...")
    
    if not os.path.exists('requirements.txt'):
        print("  Warning: requirements.txt not found. Creating basic requirements file...")
        with open('requirements.txt', 'w') as f:
            f.write("numpy\n")
            f.write("opencv-python\n")
            f.write("tensorflow>=2.4.0\n")
            f.write("pillow\n")
            f.write("matplotlib\n")
            f.write("scikit-image\n")
            f.write("tqdm\n")
            f.write("pandas\n")
            f.write("tabulate\n")
    else:
        print("  requirements.txt found.")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("  Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("  Warning: Failed to install dependencies. Please install them manually.")

def check_scripts_executable():
    """Make sure workflow scripts are executable."""
    if os.path.exists('run_workflow.sh'):
        if not os.access('run_workflow.sh', os.X_OK):
            print("\nMaking run_workflow.sh executable...")
            os.chmod('run_workflow.sh', 0o755)

def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "=" * 80)
    print("ENVIRONMENT SETUP COMPLETED")
    print("=" * 80)
    
    print("\nNext steps:")
    
    if os.path.exists('data/train/hr') and len(os.listdir('data/train/hr')) > 0:
        print("1. Train a super-resolution model:")
        print("   python train.py --model srcnn --train_dir data/train/hr --val_dir data/val/hr --epochs 100")
    else:
        print("1. Download the DIV2K dataset for training:")
        print("   python setup_environment.py --download_div2k")
    
    print("\n2. Run traditional super-resolution on a sample image:")
    print("   python inference.py --input_image data/samples/your_image.png --model bicubic --scale_factor 4")
    
    print("\n3. Try the complete workflow:")
    print("   ./run_workflow.sh")
    
    print("\n4. For a quick demo with sample images:")
    print("   python quick_demo.py")
    
    print("\nFor more examples and instructions, see QUICKSTART.md")

def main():
    """Main function to set up the environment."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("SETTING UP ENVIRONMENT FOR IMAGE SUPER-RESOLUTION & ENHANCEMENT")
    print("=" * 80)
    
    # Verify requirements
    verify_requirements()
    
    # Create directory structure
    if not args.skip_dir_creation:
        create_directory_structure()
    
    # Download datasets
    try:
        if args.download_div2k or args.download_test_only or args.datasets:
            download_datasets(args)
    except Exception as e:
        print(f"Error downloading datasets: {e}")
    
    # Create sample images
    if args.create_sample_images:
        create_sample_images()
    
    # Make sure scripts are executable
    check_scripts_executable()
    
    # Display next steps
    display_next_steps()

if __name__ == '__main__':
    main() 