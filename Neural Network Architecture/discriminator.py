"""
Discriminator Class for cGAN

This class implements the Discriminator in a Generative Adversarial Network (GAN) using TensorFlow. The Discriminator 
is responsible for distinguishing between real and generated data (fake images). The architecture consists of multiple 
downsampling layers followed by convolutional layers to extract features and make the final classification.

Key Components:
- `MNetDownsample`: A downsampling block that reduces the resolution of the input while learning spatial features.
- `Custom Upsample`: Upsampling block used for generating the output.
- `Concatenate`: Combines input and target images for joint feature extraction.
- `Conv2D layers`: Apply convolutional layers to classify the input data as real or fake.

The Discriminator is used in conjunction with a Generator in a GAN framework, where the Generator creates synthetic data 
and the Discriminator evaluates its authenticity.

"""

from tensorflow.keras import layers
import tensorflow as tf
class Discriminator(tf.keras.Model):
    def __init__(self, dis_config):
        super(Discriminator, self).__init__()
        
        # Initializer
        initializer = tf.random_normal_initializer(0., 0.02)
        
        # Concatenate layer defined here
        self.concat = layers.Concatenate()
        
        # Define the input layers for input and target images
        self.inp = layers.Input(shape=dis_config["input_shape"], name='input_image')
        self.tar = layers.Input(shape=dis_config["target_shape"], name='target_image')
        
        # Concatenate the input and target images
        x = self.concat([self.inp, self.tar])
        def downsample(filters, size, apply_batchnorm=True):
            initializer = tf.random_normal_initializer(0., 0.02)
            result = tf.keras.Sequential()
            result.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
            
            if apply_batchnorm:
                result.add(layers.BatchNormalization())
            result.add(layers.LeakyReLU())
            return result
        
        # Downsampling layers
        down1 = downsample(dis_config["base_filters"], dis_config["kernel_size"], False)(x)
        down2 = downsample(dis_config["base_filters"] * dis_config["filter_multiplier"], dis_config["kernel_size"])(down1)
        down3 = downsample(dis_config["base_filters"] * dis_config["filter_multiplier"], dis_config["kernel_size"])(down2)

        # Padding and Conv2D layer
        zero_pad1 = layers.ZeroPadding2D()(down3)
        conv = layers.Conv2D(32, 4, strides=1, use_bias=False, kernel_initializer=initializer)(zero_pad1)

        # Batch Normalization and Leaky ReLU
        batchnorm1 = layers.BatchNormalization()(conv)
        leaky_relu = layers.LeakyReLU()(batchnorm1)

        # Padding and final Conv2D layer
        zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
        last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        # Define the model
        self.model = tf.keras.Model(inputs=[self.inp, self.tar], outputs=last)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
