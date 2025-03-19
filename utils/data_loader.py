import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2
from .image_utils import (
    load_image, normalize_image, create_lr_image, 
    random_crop, random_flip, random_rotate
)

class DIV2KDataset:
    """
    Data loader for DIV2K dataset.
    DIV2K is a dataset commonly used for super-resolution tasks.
    """
    
    def __init__(self, 
                 hr_dir, 
                 scale_factor=4,
                 crop_size=96,
                 batch_size=16,
                 downgrade='bicubic',
                 augment=True):
        """
        Initialize DIV2K dataset.
        
        Args:
            hr_dir (str): Directory containing high-resolution images.
            scale_factor (int): Scale factor for downsampling.
            crop_size (int): Size of random crops for training.
            batch_size (int): Batch size.
            downgrade (str): Downgrading method. Options: 'bicubic', 'bilinear', 'nearest'.
            augment (bool): Whether to use data augmentation.
        """
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.downgrade = downgrade
        self.augment = augment
        
        # Get list of image paths
        self.hr_image_paths = self._get_image_paths(hr_dir)
        
        # Shuffle paths
        np.random.shuffle(self.hr_image_paths)
        
        # Calculate number of batches
        self.n_images = len(self.hr_image_paths)
        self.n_batches = int(np.ceil(self.n_images / self.batch_size))
    
    def _get_image_paths(self, dir_path):
        """
        Get list of image paths in directory.
        
        Args:
            dir_path (str): Directory path.
            
        Returns:
            list: List of image paths.
        """
        image_paths = []
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for ext in valid_extensions:
            image_paths.extend(list(Path(dir_path).glob(f'*{ext}')))
        
        return [str(path) for path in image_paths]
    
    def _process_image(self, hr_path):
        """
        Process a single image for training.
        
        Args:
            hr_path (str): Path to high-resolution image.
            
        Returns:
            tuple: Tuple containing (LR image, HR image).
        """
        # Load HR image
        hr_image = load_image(hr_path)
        
        # Random crop
        if hr_image.shape[0] > self.crop_size and hr_image.shape[1] > self.crop_size:
            hr_image = random_crop(hr_image, self.crop_size, self.crop_size)
        else:
            # Resize if image is too small
            hr_image = cv2.resize(hr_image, (self.crop_size, self.crop_size))
        
        # Augmentation
        if self.augment:
            hr_image = random_flip(hr_image)
            hr_image = random_rotate(hr_image)
        
        # Create LR image
        lr_image = create_lr_image(hr_image, self.scale_factor, method=self.downgrade)
        
        # Normalize
        hr_image = normalize_image(hr_image)
        lr_image = normalize_image(lr_image)
        
        return lr_image, hr_image
    
    def __len__(self):
        """
        Get number of batches.
        
        Returns:
            int: Number of batches.
        """
        return self.n_batches
    
    def __iter__(self):
        """
        Create iterator for dataset.
        
        Returns:
            self: Iterator.
        """
        self.batch_index = 0
        return self
    
    def __next__(self):
        """
        Get next batch.
        
        Returns:
            tuple: Tuple containing (LR batch, HR batch).
        """
        if self.batch_index >= self.n_batches:
            raise StopIteration
        
        # Get batch indices
        start_idx = self.batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_images)
        
        # Get batch paths
        batch_paths = self.hr_image_paths[start_idx:end_idx]
        
        # Process batch
        lr_batch = []
        hr_batch = []
        
        for path in batch_paths:
            lr_img, hr_img = self._process_image(path)
            lr_batch.append(lr_img)
            hr_batch.append(hr_img)
        
        # Increment batch index
        self.batch_index += 1
        
        return np.array(lr_batch), np.array(hr_batch)

def create_tf_dataset(hr_dir, scale_factor=4, crop_size=96, batch_size=16, 
                     downgrade='bicubic', augment=True):
    """
    Create TensorFlow dataset for super-resolution.
    
    Args:
        hr_dir (str): Directory containing high-resolution images.
        scale_factor (int): Scale factor for downsampling.
        crop_size (int): Size of random crops for training.
        batch_size (int): Batch size.
        downgrade (str): Downgrading method. Options: 'bicubic', 'bilinear', 'nearest'.
        augment (bool): Whether to use data augmentation.
        
    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    # Get list of image paths
    hr_image_paths = []
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    for ext in valid_extensions:
        hr_image_paths.extend(list(Path(hr_dir).glob(f'*{ext}')))
    
    hr_image_paths = [str(path) for path in hr_image_paths]
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(hr_image_paths)
    
    # Define preprocessing function
    def process_path(path):
        def _process_image(img_path):
            # Load HR image
            img_path = img_path.numpy().decode('utf-8')
            hr_image = load_image(img_path)
            
            # Random crop
            if hr_image.shape[0] > crop_size and hr_image.shape[1] > crop_size:
                hr_image = random_crop(hr_image, crop_size, crop_size)
            else:
                # Resize if image is too small
                hr_image = cv2.resize(hr_image, (crop_size, crop_size))
            
            # Augmentation
            if augment:
                hr_image = random_flip(hr_image)
                hr_image = random_rotate(hr_image)
            
            # Create LR image
            lr_image = create_lr_image(hr_image, scale_factor, method=downgrade)
            
            # Normalize
            hr_image = normalize_image(hr_image)
            lr_image = normalize_image(lr_image)
            
            return lr_image, hr_image
        
        # Use tf.py_function to wrap the Python function
        lr, hr = tf.py_function(_process_image, [path], [tf.float32, tf.float32])
        
        # Set shape information that gets lost in the py_function
        lr.set_shape([crop_size // scale_factor, crop_size // scale_factor, 3])
        hr.set_shape([crop_size, crop_size, 3])
        
        return lr, hr
    
    # Map preprocessing function
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def prepare_datasets(train_dir, val_dir, test_dir, **kwargs):
    """
    Prepare training, validation, and test datasets.
    
    Args:
        train_dir (str): Directory containing training images.
        val_dir (str): Directory containing validation images.
        test_dir (str): Directory containing test images.
        **kwargs: Additional arguments to pass to create_tf_dataset.
        
    Returns:
        tuple: Tuple containing (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = create_tf_dataset(train_dir, augment=True, **kwargs)
    val_dataset = create_tf_dataset(val_dir, augment=False, **kwargs)
    test_dataset = create_tf_dataset(test_dir, augment=False, **kwargs)
    
    return train_dataset, val_dataset, test_dataset 