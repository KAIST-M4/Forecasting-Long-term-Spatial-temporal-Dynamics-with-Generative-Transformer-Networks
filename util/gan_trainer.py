
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

class GANTrainer:
    def __init__(self, generator, discriminator, learning_rate=2e-4, beta_1=0.5):
        # Initialize the GAN trainer
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.beta_1 = beta_1

        # Loss functions
        self.loss_object = BinaryCrossentropy(from_logits=True)

        # Optimizers
        self.generator_optimizer = Adam(self.learning_rate, beta_1=self.beta_1)
        self.discriminator_optimizer = Adam(self.learning_rate, beta_1=self.beta_1)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (100 * l1_loss)
        return total_gen_loss

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate output from the generator
            gen_output = self.generator(input_image, training=True)

            # Get discriminator outputs
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            # Compute losses
            gen_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

            # Get gradients and apply them
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, dataset, epochs):
        gen_loss_history = []
        disc_loss_history = []

        # Iterate through epochs
        for epoch in range(epochs):
            epoch_gen_loss = []
            epoch_disc_loss = []

            # Iterate through dataset
            for input_image, target in dataset:
                gen_loss, disc_loss = self.train_step(input_image, target)
                epoch_gen_loss.append(gen_loss.numpy())
                epoch_disc_loss.append(disc_loss.numpy())

            # Store average loss per epoch
            gen_loss_history.append(np.mean(epoch_gen_loss))
            disc_loss_history.append(np.mean(epoch_disc_loss))

            # Print losses for each epoch
            print(f'Epoch {epoch + 1}, Gen Loss: {np.mean(epoch_gen_loss)}, Disc Loss: {np.mean(epoch_disc_loss)}')

        return gen_loss_history, disc_loss_history 