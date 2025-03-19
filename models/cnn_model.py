import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import PReLU, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class SRCNN:
    def __init__(self, lr_height=32, lr_width=32, channels=3, upscale_factor=4, lr=1e-4):
        """
        Initialize SRCNN model.
        
        Args:
            lr_height (int): Height of low-resolution image.
            lr_width (int): Width of low-resolution image.
            channels (int): Number of channels in the image.
            upscale_factor (int): Factor to upscale the image by.
            lr (float): Learning rate for optimizer.
        """
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.lr_shape = (lr_height, lr_width, channels)
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.hr_height = self.lr_height * self.upscale_factor
        self.hr_width = self.lr_width * self.upscale_factor
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        # Optimizer
        self.optimizer = Adam(lr)
        
        # Build model
        self.model = self.build_model()
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae', self.psnr])

    def psnr(self, y_true, y_pred):
        """
        Calculate PSNR metric.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            tf.Tensor: PSNR value.
        """
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
    def build_model(self):
        """
        Build the SRCNN model.
        
        Returns:
            tf.keras.Model: SRCNN model.
        """
        # Input layer
        input_layer = Input(shape=self.lr_shape)
        
        # Bicubic upsampling
        x = UpSampling2D(size=self.upscale_factor, interpolation='bicubic')(input_layer)
        
        # First layer - Feature extraction
        x = Conv2D(64, kernel_size=9, padding='same')(x)
        x = Activation('relu')(x)
        
        # Second layer - Non-linear mapping
        x = Conv2D(32, kernel_size=1, padding='same')(x)
        x = Activation('relu')(x)
        
        # Third layer - Reconstruction
        output_layer = Conv2D(self.channels, kernel_size=5, padding='same')(x)
        
        return Model(inputs=input_layer, outputs=output_layer)

class EDSR:
    def __init__(self, lr_height=32, lr_width=32, channels=3, upscale_factor=4, num_filters=64, 
                 num_res_blocks=16, res_scaling=0.1, lr=1e-4):
        """
        Initialize EDSR (Enhanced Deep Super-Resolution) model.
        
        Args:
            lr_height (int): Height of low-resolution image.
            lr_width (int): Width of low-resolution image.
            channels (int): Number of channels in the image.
            upscale_factor (int): Factor to upscale the image by.
            num_filters (int): Number of filters in convolutional layers.
            num_res_blocks (int): Number of residual blocks.
            res_scaling (float): Scaling factor for residual blocks.
            lr (float): Learning rate for optimizer.
        """
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.lr_shape = (lr_height, lr_width, channels)
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.hr_height = self.lr_height * self.upscale_factor
        self.hr_width = self.lr_width * self.upscale_factor
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        self.res_scaling = res_scaling
        
        # Optimizer
        self.optimizer = Adam(lr)
        
        # Build model
        self.model = self.build_model()
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mae', self.psnr, self.ssim])

    def psnr(self, y_true, y_pred):
        """
        Calculate PSNR metric.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            tf.Tensor: PSNR value.
        """
        return tf.image.psnr(y_true, y_pred, max_val=1.0)
    
    def ssim(self, y_true, y_pred):
        """
        Calculate SSIM metric.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            tf.Tensor: SSIM value.
        """
        return tf.image.ssim(y_true, y_pred, max_val=1.0)
    
    def res_block(self, x, filters, kernel_size=3, scaling=0.1):
        """
        Residual block for EDSR.
        
        Args:
            x: Input tensor.
            filters (int): Number of filters.
            kernel_size (int): Kernel size.
            scaling (float): Scaling factor for residual.
            
        Returns:
            tf.Tensor: Output tensor.
        """
        shortcut = x
        
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        
        # Scale the residual
        x = tf.multiply(x, scaling)
        
        x = Add()([shortcut, x])
        
        return x
    
    def upsample(self, x, scale_factor, num_filters):
        """
        Upsampling block for EDSR.
        
        Args:
            x: Input tensor.
            scale_factor (int): Scale factor for upsampling.
            num_filters (int): Number of filters.
            
        Returns:
            tf.Tensor: Output tensor.
        """
        x = Conv2D(num_filters * (scale_factor ** 2), 3, padding='same')(x)
        x = tf.nn.depth_to_space(x, scale_factor)  # Pixel shuffle
        x = Activation('relu')(x)
        return x
    
    def build_model(self):
        """
        Build the EDSR model.
        
        Returns:
            tf.keras.Model: EDSR model.
        """
        # Input layer
        input_layer = Input(shape=self.lr_shape)
        
        # Initial convolution
        x = Conv2D(self.num_filters, 3, padding='same')(input_layer)
        x_skip = x
        
        # Residual blocks
        for _ in range(self.num_res_blocks):
            x = self.res_block(x, self.num_filters, scaling=self.res_scaling)
        
        # Skip connection
        x = Conv2D(self.num_filters, 3, padding='same')(x)
        x = Add()([x_skip, x])
        
        # Upsampling
        if self.upscale_factor in [2, 4, 8]:
            # For factors that are powers of 2
            iterations = {2: 1, 4: 2, 8: 3}[self.upscale_factor]
            for _ in range(iterations):
                x = self.upsample(x, 2, self.num_filters)
        elif self.upscale_factor == 3:
            x = self.upsample(x, 3, self.num_filters)
        
        # Final convolution
        output_layer = Conv2D(self.channels, 3, padding='same')(x)
        
        return Model(inputs=input_layer, outputs=output_layer)

    def train(self, epochs, batch_size, train_data, val_data=None, callbacks=None):
        """
        Train the model.
        
        Args:
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            train_data: Training data generator.
            val_data: Validation data generator.
            callbacks: List of callbacks for training.
            
        Returns:
            tf.keras.callbacks.History: Training history.
        """
        return self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks
        )
    
    def save_model(self, path='edsr_model.h5'):
        """
        Save the trained model.
        
        Args:
            path (str): Path to save model.
        """
        self.model.save(path)
        
    def load_model(self, path='edsr_model.h5'):
        """
        Load trained model.
        
        Args:
            path (str): Path to model.
        """
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                'psnr': self.psnr,
                'ssim': self.ssim
            }
        ) 