import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Dense
from tensorflow.keras.layers import Flatten, PReLU, UpSampling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam

class SRGAN:
    def __init__(self, 
                 lr_height=32, 
                 lr_width=32, 
                 channels=3, 
                 upscale_factor=4, 
                 gen_lr=1e-4, 
                 dis_lr=1e-4):
        """
        Initialize SRGAN model.
        
        Args:
            lr_height (int): Height of low-resolution image.
            lr_width (int): Width of low-resolution image.
            channels (int): Number of channels in the image.
            upscale_factor (int): Factor to upscale the image by.
            gen_lr (float): Learning rate for generator.
            dis_lr (float): Learning rate for discriminator.
        """
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.lr_shape = (lr_height, lr_width, channels)
        self.channels = channels
        self.upscale_factor = upscale_factor
        self.hr_height = self.lr_height * self.upscale_factor
        self.hr_width = self.lr_width * self.upscale_factor
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        # Optimizers
        self.gen_optimizer = Adam(gen_lr, 0.9)
        self.dis_optimizer = Adam(dis_lr, 0.9)
        
        # Build and compile the discriminator network
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.dis_optimizer,
            metrics=['accuracy']
        )
        
        # Build the generator network
        self.generator = self.build_generator()
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # HR image input
        hr_input = Input(shape=self.hr_shape)
        
        # LR image input
        lr_input = Input(shape=self.lr_shape)
        
        # Generate HR image from LR
        fake_hr = self.generator(lr_input)
        
        # Discriminator determines validity of generated HR images
        validity = self.discriminator(fake_hr)
        
        # VGG model for perceptual loss
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        
        # Get VGG features
        vgg_hr = self.vgg(hr_input)
        vgg_sr = self.vgg(fake_hr)
        
        # Combined model (generator + discriminator + VGG)
        self.combined = Model([lr_input, hr_input], [validity, vgg_sr, vgg_hr])
        self.combined.compile(
            loss=['binary_crossentropy', 'mse', 'mse'],
            loss_weights=[1e-3, 6e-3, 1.0],
            optimizer=self.gen_optimizer
        )

    def build_vgg(self):
        """
        Build VGG19 model for perceptual loss.
        
        Returns:
            tf.keras.Model: VGG19 model.
        """
        vgg = VGG19(weights='imagenet', include_top=False, input_shape=self.hr_shape)
        
        # Extract features from specific layer
        # We use the 9th layer for perceptual loss
        return Model(inputs=vgg.input, outputs=vgg.layers[9].output)

    def residual_block(self, x, filters, kernel_size=3, strides=1, padding='same'):
        """
        Residual block for generator.
        
        Args:
            x: Input tensor.
            filters (int): Number of filters.
            kernel_size (int): Kernel size.
            strides (int): Strides.
            padding (str): Padding type.
            
        Returns:
            tf.Tensor: Output tensor.
        """
        shortcut = x
        
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU(shared_axes=[1, 2])(x)
        
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization(momentum=0.8)(x)
        
        x = Add()([shortcut, x])
        
        return x

    def build_generator(self):
        """
        Build the generator network.
        
        Returns:
            tf.keras.Model: Generator model.
        """
        # Input layer
        input_layer = Input(shape=self.lr_shape)
        
        # Pre-residual block
        x = Conv2D(64, 9, padding='same')(input_layer)
        x = PReLU(shared_axes=[1, 2])(x)
        
        # Store the output of the pre-residual block
        pre_res = x
        
        # Residual blocks
        for _ in range(16):
            x = self.residual_block(x, 64)
        
        # Post-residual block
        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Add()([pre_res, x])
        
        # Upsampling blocks
        for _ in range(2):  # For 4x upscaling we need 2 upsampling blocks (2^2 = 4)
            x = Conv2D(256, 3, padding='same')(x)
            x = UpSampling2D(size=2)(x)
            x = PReLU(shared_axes=[1, 2])(x)
        
        # Output layer
        output_layer = Conv2D(self.channels, 9, padding='same', activation='tanh')(x)
        
        return Model(inputs=input_layer, outputs=output_layer)

    def build_discriminator(self):
        """
        Build the discriminator network.
        
        Returns:
            tf.keras.Model: Discriminator model.
        """
        # Input layer
        input_layer = Input(shape=self.hr_shape)
        
        # First conv block
        x = Conv2D(64, 3, strides=1, padding='same')(input_layer)
        x = LeakyReLU(alpha=0.2)(x)
        
        # Conv blocks with increasing filters
        filter_sizes = [64, 128, 128, 256, 256, 512, 512]
        strides = [2, 1, 2, 1, 2, 1, 2]
        
        for f, s in zip(filter_sizes, strides):
            x = Conv2D(f, 3, strides=s, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
        
        # Dense layers
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=input_layer, outputs=output_layer)

    def train(self, epochs, batch_size, train_data):
        """
        Train the SRGAN model.
        
        Args:
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            train_data: Training data generator.
            
        Returns:
            dict: Training history.
        """
        # Labels for adversarial loss
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        history = {'d_loss': [], 'g_loss': [], 'd_acc': []}
        
        for epoch in range(epochs):
            d_loss_epoch = []
            g_loss_epoch = []
            d_acc_epoch = []
            
            for batch_i, (lr_imgs, hr_imgs) in enumerate(train_data):
                # Generate fake HR images
                fake_hr = self.generator.predict(lr_imgs)
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(hr_imgs, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake_labels)
                d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)
                
                # Train generator
                g_loss = self.combined.train_on_batch([lr_imgs, hr_imgs], 
                                                    [real_labels, 
                                                     self.vgg.predict(hr_imgs), 
                                                     self.vgg.predict(hr_imgs)])
                
                # Append batch loss
                d_loss_epoch.append(d_loss[0])
                g_loss_epoch.append(g_loss[0])
                d_acc_epoch.append(d_loss[1])
                
                # Print batch progress
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_i+1}/{len(train_data)}, "
                      f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss[0]:.4f}")
            
            # Append epoch average loss
            history['d_loss'].append(tf.reduce_mean(d_loss_epoch).numpy())
            history['g_loss'].append(tf.reduce_mean(g_loss_epoch).numpy())
            history['d_acc'].append(tf.reduce_mean(d_acc_epoch).numpy())
            
            # Print epoch progress
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"D Loss: {history['d_loss'][-1]:.4f}, D Acc: {history['d_acc'][-1]:.4f}, "
                  f"G Loss: {history['g_loss'][-1]:.4f}")
            
        return history

    def save_models(self, gen_path='generator.h5', dis_path='discriminator.h5'):
        """
        Save the trained models.
        
        Args:
            gen_path (str): Path to save generator model.
            dis_path (str): Path to save discriminator model.
        """
        self.generator.save(gen_path)
        self.discriminator.save(dis_path)
        
    def load_models(self, gen_path='generator.h5', dis_path='discriminator.h5'):
        """
        Load trained models.
        
        Args:
            gen_path (str): Path to generator model.
            dis_path (str): Path to discriminator model.
        """
        self.generator = tf.keras.models.load_model(gen_path)
        self.discriminator = tf.keras.models.load_model(dis_path) 