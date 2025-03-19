# Image Super-Resolution & Enhancement

This project implements and compares various image super-resolution and enhancement techniques, including both traditional methods and deep learning approaches (SRGAN, CNN-based models).

## Project Overview

Super-resolution (SR) is the process of recovering high-resolution (HR) images from low-resolution (LR) images. This project:

1. Implements traditional upscaling methods (bicubic, bilinear, Lanczos)
2. Implements deep learning-based approaches (SRGAN and CNN models)
3. Provides tools for visual and quantitative comparison between methods
4. Includes training and evaluation pipelines

## Project Structure

```
├── data/               # Directory for datasets
│   ├── train/          # Training images
│   ├── val/            # Validation images  
│   └── test/           # Test images
├── models/             # Model implementations
│   ├── traditional.py  # Traditional upscaling methods
│   ├── srgan.py        # SRGAN implementation
│   └── cnn_model.py    # CNN-based model
├── utils/              # Utility functions
│   ├── data_loader.py  # Data loading utilities
│   ├── metrics.py      # Evaluation metrics (PSNR, SSIM)
│   └── image_utils.py  # Image processing utilities
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── compare.py          # Comparison between methods
├── inference.py        # Inference script for super-resolution
└── requirements.txt    # Project dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/lehuyqb/Image-Super-Resolution-Enhancement.git
cd Image-Super-Resolution-Enhancement
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the SRGAN model:

```bash
python train.py --model srgan --epochs 100 --batch_size 16 --data_dir data/train
```

To train a CNN model:

```bash
python train.py --model cnn --epochs 100 --batch_size 16 --data_dir data/train
```

### Evaluation

Evaluate the model performance:

```bash
python evaluate.py --model_path saved_models/srgan_model.h5 --test_dir data/test
```

### Comparison

Compare different super-resolution methods:

```bash
python compare.py --input_image path/to/image.jpg --scale 4 --methods bicubic srgan cnn
```

### Inference

Apply super-resolution to your own images:

```bash
python inference.py --input_image path/to/image.jpg --model srgan --output_path enhanced_image.png
```

## Results

The project includes a comprehensive comparison between traditional and deep learning-based approaches:

- Quantitative comparison using PSNR, SSIM, and perceptual metrics
- Visual comparison of the results
- Analysis of computational requirements and processing time

## License

MIT License 