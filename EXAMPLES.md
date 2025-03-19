# Image Super-Resolution & Enhancement: Common Examples

This document provides examples of common tasks and commands for the Image Super-Resolution & Enhancement project.

## Quick Start Examples

### Setting Up the Environment

```bash
# Basic setup (create directories only)
python setup_environment.py

# Setup and download test datasets (Set5 and Set14)
python setup_environment.py --download_test_only

# Setup and download everything (DIV2K + test datasets)
python setup_environment.py --download_div2k --datasets all
```

### Running Quick Demo

```bash
# Run a quick demo with default settings
python quick_demo.py

# Run demo with a specific image
python quick_demo.py --input_image your_image.jpg

# Run demo with specific methods and scale factor
python quick_demo.py --methods bicubic lanczos srgan --scale_factor 2
```

### Running Complete Workflow

```bash
# Run the complete workflow script
./run_workflow.sh
```

## Training Examples

### Training SRCNN

```bash
# Basic SRCNN training
python train.py --model srcnn --train_dir data/train/hr --val_dir data/val/hr --epochs 100

# SRCNN with custom parameters
python train.py --model srcnn --train_dir data/train/hr --val_dir data/val/hr \
                --epochs 200 --batch_size 32 --scale_factor 4 \
                --learning_rate 1e-4 --crop_size 128
```

### Training EDSR

```bash
# Basic EDSR training
python train.py --model edsr --train_dir data/train/hr --val_dir data/val/hr --epochs 100

# EDSR with custom parameters
python train.py --model edsr --train_dir data/train/hr --val_dir data/val/hr \
                --epochs 300 --batch_size 16 --scale_factor 2 \
                --learning_rate 5e-5 --crop_size 96
```

### Training SRGAN

```bash
# Basic SRGAN training
python train.py --model srgan --train_dir data/train/hr --val_dir data/val/hr --epochs 100

# SRGAN with custom parameters
python train.py --model srgan --train_dir data/train/hr --val_dir data/val/hr \
                --epochs 200 --batch_size 8 --scale_factor 4 \
                --learning_rate 1e-4 --crop_size 96
```

## Evaluation Examples

### Evaluating Traditional Methods

```bash
# Evaluate bicubic upscaling
python evaluate.py --model_type traditional --traditional_method bicubic \
                  --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate lanczos upscaling
python evaluate.py --model_type traditional --traditional_method lanczos \
                  --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate all traditional methods on multiple test sets
for method in bicubic bilinear lanczos nearest; do
  for test_set in data/test/set5 data/test/set14; do
    python evaluate.py --model_type traditional --traditional_method $method \
                      --test_dir $test_set --scale_factor 4 --save_images
  done
done
```

### Evaluating Deep Learning Models

```bash
# Evaluate SRCNN
python evaluate.py --model_type srcnn --model_path saved_models/srcnn_final_model.h5 \
                  --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate EDSR
python evaluate.py --model_type edsr --model_path saved_models/edsr_final_model.h5 \
                  --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate SRGAN
python evaluate.py --model_type srgan --model_path saved_models/srgan_generator_final_model.h5 \
                  --test_dir data/test/set5 --scale_factor 4 --save_images
```

## Comparison Examples

```bash
# Compare traditional methods
python compare.py --input_image data/test/set5/baby.png \
                 --methods bicubic bilinear lanczos \
                 --scale_factor 4 --output_dir results/comparison

# Compare deep learning models with traditional methods
python compare.py --input_image data/test/set5/baby.png \
                 --methods bicubic srcnn edsr srgan \
                 --srcnn_model saved_models/srcnn_final_model.h5 \
                 --edsr_model saved_models/edsr_final_model.h5 \
                 --srgan_model saved_models/srgan_generator_final_model.h5 \
                 --scale_factor 4 --output_dir results/comparison
```

## Inference Examples

```bash
# Run inference with bicubic upscaling
python inference.py --input_image your_image.jpg --model bicubic \
                   --scale_factor 4 --show_result

# Run inference with SRCNN
python inference.py --input_image your_image.jpg --model srcnn \
                   --model_path saved_models/srcnn_final_model.h5 \
                   --scale_factor 4 --show_result

# Run inference with SRGAN and compare with original
python inference.py --input_image your_low_res.jpg --model srgan \
                   --model_path saved_models/srgan_generator_final_model.h5 \
                   --scale_factor 4 --show_result \
                   --compare_with_original --original_image your_high_res.jpg
```

## Dataset Examples

```bash
# Download DIV2K dataset (training)
python download_dataset.py --dataset div2k --output_dir data --scale_factor 4

# Download Set5 test dataset
python download_dataset.py --dataset set5 --output_dir data

# Download multiple test datasets
python download_dataset.py --dataset set5 set14 urban100 --output_dir data
```

## Advanced Usage Examples

### Using Pre-trained Models

```bash
# Download pre-trained models (if available)
wget -P saved_models https://example.com/path/to/pretrained/srcnn_model.h5
wget -P saved_models https://example.com/path/to/pretrained/srgan_generator.h5

# Run inference with pre-trained model
python inference.py --input_image your_image.jpg --model srcnn \
                   --model_path saved_models/srcnn_model.h5 \
                   --scale_factor 4 --show_result
```

### Creating GIFs of Results

```bash
# Create a GIF comparing different methods (requires ImageMagick)
mkdir -p results/gif
python compare.py --input_image your_image.jpg --methods bicubic lanczos srgan \
                 --srgan_model saved_models/srgan_generator_final_model.h5 \
                 --scale_factor 4 --output_dir results/gif
convert -delay 100 results/gif/*.png results/comparison.gif
```

### Batch Processing

```bash
# Process multiple images
for img in your_images/*.jpg; do
  python inference.py --input_image $img --model srcnn \
                     --model_path saved_models/srcnn_final_model.h5 \
                     --scale_factor 4
done
```

### Custom Training Loop

```bash
# Training with custom settings for specific hardware
python train.py --model srcnn --train_dir data/train/hr --val_dir data/val/hr \
                --epochs 100 --batch_size 8 --crop_size 64 --learning_rate 1e-4 \
                --save_every 10 --samples_dir results/training_samples
```

## Troubleshooting

If you encounter any issues, try the following:

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Check if the required directories exist:
   ```bash
   python setup_environment.py
   ```

3. For GPU out-of-memory errors, reduce batch size or crop size:
   ```bash
   python train.py --model srgan --batch_size 4 --crop_size 64
   ```

4. For CPU-only training, use smaller models and reduce batch size:
   ```bash
   python train.py --model srcnn --batch_size 4
   ``` 