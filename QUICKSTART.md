# Quickstart Guide for Image Super-Resolution & Enhancement

This guide will help you get started with the Image Super-Resolution & Enhancement project. It provides practical examples for downloading datasets, training models, evaluating performance, and applying super-resolution to your own images.

## Setting Up the Environment

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Image-Super-Resolution-Enhancement.git
cd Image-Super-Resolution-Enhancement
pip install -r requirements.txt
```

2. Create the necessary directories:

```bash
mkdir -p data/train data/val data/test saved_models samples results
```

## Downloading Datasets

The project includes a script to download datasets commonly used for super-resolution tasks:

### Downloading DIV2K Training Dataset

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is a high-quality dataset specifically designed for super-resolution tasks:

```bash
# Download DIV2K dataset with scale factor x4
python download_dataset.py --dataset div2k --div2k_scale 4 --output_dir data
```

### Downloading Test Datasets

Several standard benchmark datasets are available:

```bash
# Download Set5 test dataset
python download_dataset.py --dataset set5 --output_dir data

# Download Set14 test dataset
python download_dataset.py --dataset set14 --output_dir data

# Download Urban100 test dataset
python download_dataset.py --dataset urban100 --output_dir data
```

## Quick Test: Traditional Methods

You can quickly test traditional upscaling methods without training:

```bash
# Apply bicubic upscaling to an image
python inference.py --input_image data/test/set5/baby.png --model bicubic --scale_factor 4 --show_result

# Apply Lanczos upscaling to an image
python inference.py --input_image data/test/set5/baby.png --model lanczos --scale_factor 4 --show_result
```

## Training Models

### Training SRCNN Model

SRCNN is a relatively simple CNN-based super-resolution model:

```bash
python train.py --model srcnn --train_dir data/train/hr --val_dir data/val/hr --epochs 50 --batch_size 16 --scale_factor 4 --crop_size 96 --learning_rate 1e-4
```

### Training EDSR Model

EDSR (Enhanced Deep Super-Resolution) is a more powerful model:

```bash
python train.py --model edsr --train_dir data/train/hr --val_dir data/val/hr --epochs 100 --batch_size 16 --scale_factor 4 --crop_size 128 --learning_rate 1e-4
```

### Training SRGAN Model

SRGAN uses a generative adversarial network approach:

```bash
python train.py --model srgan --train_dir data/train/hr --val_dir data/val/hr --epochs 200 --batch_size 16 --scale_factor 4 --crop_size 96 --learning_rate 1e-4
```

## Evaluating Models

Evaluate the performance of traditional methods:

```bash
python evaluate.py --model_type traditional --traditional_method bicubic --test_dir data/test/set5 --scale_factor 4 --save_images
```

Evaluate trained deep learning models:

```bash
# Evaluate SRCNN model
python evaluate.py --model_type srcnn --model_path saved_models/srcnn_final_model.h5 --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate EDSR model
python evaluate.py --model_type edsr --model_path saved_models/edsr_final_model.h5 --test_dir data/test/set5 --scale_factor 4 --save_images

# Evaluate SRGAN model
python evaluate.py --model_type srgan --model_path saved_models/srgan_generator_epoch_200.h5 --test_dir data/test/set5 --scale_factor 4 --save_images
```

## Comparing Different Methods

Compare multiple super-resolution methods on a single image:

```bash
python compare.py --input_image data/test/set5/baby.png --scale_factor 4 --methods bicubic lanczos srcnn edsr srgan --srcnn_model saved_models/srcnn_final_model.h5 --edsr_model saved_models/edsr_final_model.h5 --srgan_model saved_models/srgan_generator_epoch_200.h5
```

## Applying Super-Resolution to Your Own Images

Apply super-resolution to your own images:

```bash
# Using traditional methods
python inference.py --input_image your_image.jpg --model bicubic --scale_factor 4

# Using SRCNN
python inference.py --input_image your_image.jpg --model srcnn --model_path saved_models/srcnn_final_model.h5 --scale_factor 4

# Using EDSR
python inference.py --input_image your_image.jpg --model edsr --model_path saved_models/edsr_final_model.h5 --scale_factor 4

# Using SRGAN
python inference.py --input_image your_image.jpg --model srgan --model_path saved_models/srgan_generator_epoch_200.h5 --scale_factor 4
```

## Tips for Best Results

1. **Training data quality matters**: Higher quality training images will produce better results.

2. **Longer training times**: Deep learning models, particularly SRGAN, benefit from longer training periods.

3. **Crop size and batch size**: 
   - If you have limited GPU memory, reduce batch size and crop size
   - For better quality, use larger crop sizes (128 or 256)

4. **Scale factors**: 
   - Start with smaller scale factors (x2, x3) for better results
   - x4 is standard but more challenging
   - x8 is extremely challenging and may require specialized architectures

5. **Pretrained models**: Consider using pretrained models (like VGG) for feature extraction or perceptual loss.

## Troubleshooting

1. **Out of memory errors**: Reduce batch size or crop size.

2. **Slow training**: Consider using a smaller subset of DIV2K for initial experiments.

3. **Poor results**: 
   - Check if your training data is appropriate for the task
   - Try increasing the model capacity (more filters/layers)
   - For SRGAN, adjust the balance between adversarial and content losses 