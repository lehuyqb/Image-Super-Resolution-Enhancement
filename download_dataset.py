import os
import argparse
import requests
import zipfile
import shutil
from tqdm import tqdm
import urllib.request

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download Super-Resolution Datasets')
    
    parser.add_argument('--dataset', type=str, default='div2k', 
                        choices=['div2k', 'set5', 'set14', 'urban100', 'manga109'],
                        help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the downloaded dataset')
    parser.add_argument('--div2k_scale', type=int, default=4,
                        choices=[2, 3, 4, 8],
                        help='Scale factor for DIV2K dataset')
    
    return parser.parse_args()

def download_file(url, output_path):
    """Download a file with progress bar."""
    print(f"Downloading from {url} to {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download with progress bar
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, 
            output_path, 
            reporthook=lambda b, bsize, tsize: t.update(bsize if tsize == -1 else (b * bsize - t.n))
        )

def extract_zip(zip_path, extract_dir):
    """Extract a zip file."""
    print(f"Extracting {zip_path} to {extract_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Create a progress bar
        total = len(zip_ref.namelist())
        with tqdm(total=total, desc="Extracting files") as pbar:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_dir)
                pbar.update(1)

def download_div2k(output_dir, scale_factor):
    """Download DIV2K dataset."""
    print(f"Downloading DIV2K dataset with scale factor x{scale_factor}")
    
    # Create dataset directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    
    # URLs for DIV2K dataset
    urls = {
        'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        f'train_lr_x{scale_factor}': f'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X{scale_factor}.zip',
        f'valid_lr_x{scale_factor}': f'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X{scale_factor}.zip'
    }
    
    # Download and extract each file
    for name, url in urls.items():
        zip_path = os.path.join(output_dir, f'{name}.zip')
        
        # Download
        download_file(url, zip_path)
        
        # Extract
        extract_dir = os.path.join(output_dir, 'temp')
        extract_zip(zip_path, extract_dir)
        
        # Move files to appropriate folders
        if 'train' in name:
            if 'hr' in name.lower():
                dest_dir = os.path.join(output_dir, 'train', 'hr')
            else:
                dest_dir = os.path.join(output_dir, 'train', f'lr_x{scale_factor}')
        else:  # validation
            if 'hr' in name.lower():
                dest_dir = os.path.join(output_dir, 'val', 'hr')
            else:
                dest_dir = os.path.join(output_dir, 'val', f'lr_x{scale_factor}')
        
        os.makedirs(dest_dir, exist_ok=True)
        
        # Find the extracted directory
        for root, dirs, files in os.walk(extract_dir):
            if files and any(f.endswith(('.png', '.jpg', '.jpeg', '.bmp')) for f in files):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(dest_dir, file)
                        shutil.move(src_path, dst_path)
        
        # Clean up
        shutil.rmtree(extract_dir, ignore_errors=True)
        os.remove(zip_path)
    
    print(f"DIV2K dataset downloaded to {output_dir}")

def download_test_dataset(dataset, output_dir):
    """Download test datasets."""
    print(f"Downloading {dataset} test dataset")
    
    # Create test directory
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    # URLs for test datasets
    urls = {
        'set5': 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip',
        'set14': 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip',
        'urban100': 'https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip',
        'manga109': 'https://uofi.box.com/shared/static/qgctsplb8txrksm9to9tayqte94oitsa.zip'
    }
    
    # Download and extract
    url = urls[dataset]
    zip_path = os.path.join(test_dir, f'{dataset}.zip')
    
    # Download
    download_file(url, zip_path)
    
    # Extract
    dataset_dir = os.path.join(test_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    extract_zip(zip_path, dataset_dir)
    
    # Clean up
    os.remove(zip_path)
    
    print(f"{dataset} test dataset downloaded to {dataset_dir}")

def main():
    args = parse_args()
    
    if args.dataset == 'div2k':
        download_div2k(args.output_dir, args.div2k_scale)
    else:
        download_test_dataset(args.dataset, args.output_dir)
    
    print("Download completed successfully!")

if __name__ == '__main__':
    main() 