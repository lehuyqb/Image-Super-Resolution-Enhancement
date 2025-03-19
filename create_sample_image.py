import numpy as np
import cv2
import os

def create_sample_image(filename, size=(256, 256)):
    """Create a simple test image."""
    # Create a gradient image
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    
    # Create RGB channels
    r = np.sin(xx * 10) * 255
    g = np.cos(yy * 10) * 255
    b = (xx + yy) * 128
    
    # Combine channels
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:,:,0] = r.astype(np.uint8)
    img[:,:,1] = g.astype(np.uint8)
    img[:,:,2] = b.astype(np.uint8)
    
    # Add a small square in the center
    center = (size[0] // 2, size[1] // 2)
    square_size = min(size) // 8
    img[center[0]-square_size:center[0]+square_size, 
        center[1]-square_size:center[1]+square_size, :] = [255, 0, 0]
    
    # Save the image
    cv2.imwrite(filename, img)
    print(f"Created sample image: {filename}")

def create_set5_samples():
    """Create sample images for the Set5 dataset."""
    # Create directory if it doesn't exist
    set5_dir = "data/test/set5"
    os.makedirs(set5_dir, exist_ok=True)
    
    # Create sample images with different sizes
    create_sample_image(os.path.join(set5_dir, "baby.png"), size=(256, 256))
    create_sample_image(os.path.join(set5_dir, "bird.png"), size=(288, 288))
    create_sample_image(os.path.join(set5_dir, "butterfly.png"), size=(320, 320))
    create_sample_image(os.path.join(set5_dir, "head.png"), size=(224, 224))
    create_sample_image(os.path.join(set5_dir, "woman.png"), size=(384, 384))
    
    # Also create samples for testing
    samples_dir = "data/samples"
    os.makedirs(samples_dir, exist_ok=True)
    create_sample_image(os.path.join(samples_dir, "test_sample.png"), size=(512, 512))
    
    print("Successfully created all sample images.")

if __name__ == "__main__":
    print("Creating sample images for testing...")
    create_set5_samples() 