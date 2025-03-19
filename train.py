import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from models.srgan import SRGAN
from models.cnn_model import SRCNN, EDSR
from utils.data_loader import prepare_datasets, create_tf_dataset
from utils.image_utils import load_image, normalize_image, create_lr_image, save_image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Super-Resolution Models')
    
    parser.add_argument('--model', type=str, default='edsr', choices=['srgan', 'srcnn', 'edsr'],
                        help='Model to train: srgan, srcnn, or edsr')
    parser.add_argument('--train_dir', type=str, default='data/train',
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default='data/val',
                        help='Directory containing validation images')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Directory containing test images')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Scale factor for super-resolution')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr_height', type=int, default=32,
                        help='Height of low-resolution image')
    parser.add_argument('--lr_width', type=int, default=32,
                        help='Width of low-resolution image')
    parser.add_argument('--crop_size', type=int, default=128,
                        help='Size of random crops for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='Directory to save trained models')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory to save sample images during training')
    
    return parser.parse_args()

def save_sample_images(model, epoch, lr_images, hr_images, sample_dir, model_type, scale_factor):
    """Save sample images during training."""
    os.makedirs(sample_dir, exist_ok=True)
    
    # Generate SR images
    if model_type == 'srgan':
        sr_images = model.generator.predict(lr_images)
    else:
        sr_images = model.model.predict(lr_images)
    
    # Save images (4 samples)
    num_samples = min(4, len(lr_images))
    
    for i in range(num_samples):
        # Create comparative figure
        plt.figure(figsize=(15, 5))
        
        # Plot LR image
        plt.subplot(1, 3, 1)
        plt.imshow(lr_images[i])
        plt.title('Low Resolution')
        plt.axis('off')
        
        # Plot SR image
        plt.subplot(1, 3, 2)
        plt.imshow(np.clip(sr_images[i], 0, 1))
        plt.title('Super Resolution')
        plt.axis('off')
        
        # Plot HR image
        plt.subplot(1, 3, 3)
        plt.imshow(hr_images[i])
        plt.title('High Resolution (Ground Truth)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{sample_dir}/{model_type}_epoch_{epoch+1}_sample_{i+1}.png')
        plt.close()

def train_srgan(args):
    """Train SRGAN model."""
    print("Preparing datasets...")
    
    # Create datasets
    train_dataset = create_tf_dataset(
        args.train_dir, 
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        batch_size=args.batch_size
    )
    
    val_dataset = create_tf_dataset(
        args.val_dir, 
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Initialize model
    print("Initializing SRGAN model...")
    srgan = SRGAN(
        lr_height=args.lr_height,
        lr_width=args.lr_width,
        upscale_factor=args.scale_factor,
        gen_lr=args.learning_rate,
        dis_lr=args.learning_rate
    )
    
    # Create directories for saving models and samples
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Get some samples for visualization
    for lr_batch, hr_batch in val_dataset.take(1):
        sample_lr = lr_batch.numpy()
        sample_hr = hr_batch.numpy()
        break
    
    print(f"Starting SRGAN training for {args.epochs} epochs...")
    
    # Train SRGAN
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train on batches
        for batch_i, (lr_imgs, hr_imgs) in enumerate(train_dataset):
            # Train discriminator
            fake_hr = srgan.generator.predict(lr_imgs)
            d_loss_real = srgan.discriminator.train_on_batch(hr_imgs, tf.ones((len(hr_imgs), 1)))
            d_loss_fake = srgan.discriminator.train_on_batch(fake_hr, tf.zeros((len(fake_hr), 1)))
            d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
            
            # Train generator
            g_loss = srgan.combined.train_on_batch(
                [lr_imgs, hr_imgs],
                [tf.ones((len(lr_imgs), 1)), 
                 srgan.vgg.predict(hr_imgs), 
                 srgan.vgg.predict(hr_imgs)]
            )
            
            # Print progress
            if batch_i % 10 == 0:
                print(f"  Batch {batch_i+1}/{len(train_dataset)}, "
                      f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss[0]:.4f}")
        
        # Save sample images
        if (epoch+1) % 5 == 0 or epoch == 0:
            save_sample_images(
                srgan, epoch, sample_lr, sample_hr, 
                args.sample_dir, 'srgan', args.scale_factor
            )
        
        # Save models periodically
        if (epoch+1) % 20 == 0 or epoch == args.epochs-1:
            srgan.save_models(
                gen_path=f'{args.save_dir}/srgan_generator_epoch_{epoch+1}.h5',
                dis_path=f'{args.save_dir}/srgan_discriminator_epoch_{epoch+1}.h5'
            )
    
    print("SRGAN training completed.")

def train_cnn(args):
    """Train CNN-based models (SRCNN or EDSR)."""
    print("Preparing datasets...")
    
    # Create datasets
    train_dataset = create_tf_dataset(
        args.train_dir, 
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        batch_size=args.batch_size
    )
    
    val_dataset = create_tf_dataset(
        args.val_dir, 
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        augment=False
    )
    
    # Initialize model
    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'srcnn':
        model = SRCNN(
            lr_height=args.lr_height,
            lr_width=args.lr_width,
            upscale_factor=args.scale_factor,
            lr=args.learning_rate
        )
    else:  # EDSR
        model = EDSR(
            lr_height=args.lr_height,
            lr_width=args.lr_width,
            upscale_factor=args.scale_factor,
            lr=args.learning_rate
        )
    
    # Create directories for saving models and samples
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Get some samples for visualization
    for lr_batch, hr_batch in val_dataset.take(1):
        sample_lr = lr_batch.numpy()
        sample_hr = hr_batch.numpy()
        break
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=f'{args.save_dir}/{args.model}_model_epoch_{{epoch:02d}}.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_freq='epoch',
            period=20
        ),
        TensorBoard(log_dir=f'logs/{args.model}'),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: save_sample_images(
                model, epoch, sample_lr, sample_hr, 
                args.sample_dir, args.model, args.scale_factor
            ) if (epoch+1) % 5 == 0 or epoch == 0 else None
        )
    ]
    
    print(f"Starting {args.model.upper()} training for {args.epochs} epochs...")
    
    # Train model
    history = model.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_data=train_dataset,
        val_data=val_dataset,
        callbacks=callbacks
    )
    
    # Save final model
    model.save_model(f'{args.save_dir}/{args.model}_final_model.h5')
    
    print(f"{args.model.upper()} training completed.")
    
    return history

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the selected model
    if args.model == 'srgan':
        train_srgan(args)
    else:  # CNN models (SRCNN or EDSR)
        history = train_cnn(args)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot PSNR
        plt.subplot(1, 2, 2)
        plt.plot(history.history['psnr'], label='Training PSNR')
        plt.plot(history.history['val_psnr'], label='Validation PSNR')
        plt.title('PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{args.sample_dir}/{args.model}_training_history.png')

if __name__ == '__main__':
    main() 