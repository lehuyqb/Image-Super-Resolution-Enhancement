import os
import urllib.request
import ssl

def create_test_directories():
    """Create test directories for the workflow."""
    print("Creating test directories...")
    
    # Create directory structure
    directories = [
        'data/test/set5',
        'data/samples',
        'results/comparison',
        'results/evaluation',
        'saved_models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created {directory}")

def download_sample_images():
    """Download sample test images directly."""
    print("\nDownloading sample test images...")
    
    # Create SSL context to handle certificate issues
    ssl_context = ssl._create_unverified_context()
    
    # Sample images URLs (these are public domain test images)
    sample_images = [
        ('https://raw.githubusercontent.com/nuhil/Super-Resolution-with-CNNs-and-GANs/master/dataset/test/Set5/baby.png', 'data/test/set5/baby.png'),
        ('https://raw.githubusercontent.com/nuhil/Super-Resolution-with-CNNs-and-GANs/master/dataset/test/Set5/bird.png', 'data/test/set5/bird.png'),
        ('https://raw.githubusercontent.com/nuhil/Super-Resolution-with-CNNs-and-GANs/master/dataset/test/Set5/butterfly.png', 'data/test/set5/butterfly.png')
    ]
    
    for url, path in sample_images:
        try:
            print(f"  Downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"  Saved to {path}")
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
    
    print("Download completed.")

def main():
    """Main function."""
    print("=" * 80)
    print("SETTING UP TEST DATA FOR IMAGE SUPER-RESOLUTION WORKFLOW")
    print("=" * 80)
    
    # Create directories
    create_test_directories()
    
    # Download sample images
    download_sample_images()
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETED")
    print("=" * 80)
    print("\nYou can now run the workflow script:")
    print("./run_workflow.sh")

if __name__ == "__main__":
    main()