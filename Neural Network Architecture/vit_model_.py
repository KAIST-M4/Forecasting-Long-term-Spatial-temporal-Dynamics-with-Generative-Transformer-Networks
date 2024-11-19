


import tensorflow as tf
from vit_config import ViTConfig
from tensorflow.keras.layers import Layer, Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization, Conv2D, ConvLSTM2D

# Transformer Block
class TransformerBlock(Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-Head Attention layer
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        
        # Feed-forward network with two Dense layers
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(key_dim * num_heads),
            ]
        )
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout layers to prevent overfitting
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training, mask=None):
        # Multi-Head Attention mechanism
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        # Apply dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection followed by layer normalization
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Spatiotemporal Vision Transformer (ViT) Model
class SpatiotemporalViT(tf.keras.Model):
    def __init__(self, config: ViTConfig):
        super(SpatiotemporalViT, self).__init__()
        
        # Initialize model parameters from config
        self.max_patches = config.max_patches
        self.hidden_size = config.hidden_size
        self.patch_shape = config.patch_shape  # (patch_size, patch_size, channels)
        self.patch_size = config.patch_size

        # Patch embedding to convert image patches into embeddings
        self.patch_embedding = tf.keras.layers.Dense(self.hidden_size)
        
        # Position embedding to encode positional information
        self.position_embedding = tf.keras.layers.Embedding(input_dim=self.max_patches, output_dim=self.hidden_size)

        # Define the transformer blocks
        self.transformer_layers = [
            TransformerBlock(
                num_heads=config.num_heads,
                key_dim=self.hidden_size // config.num_heads,
                ff_dim=config.mlp_dim,
                dropout=0.1
            )
            for _ in range(config.num_layers)
        ]
        
        # Regression head to predict the output size
        self.regression_head = tf.keras.layers.Dense(self.patch_size * self.patch_size * self.patch_shape[2])  # Output: 16*16*128 = 32768

        # ConvLSTM2D to capture temporal dependencies
        self.conv_lstm = tf.keras.layers.ConvLSTM2D(filters=self.patch_shape[2], kernel_size=(3, 3), padding="same", return_sequences=False)
        
        # Final convolutional layer
        self.conv_output = tf.keras.layers.Conv2D(filters=self.patch_shape[2], kernel_size=(3, 3), padding="same")

    def call(self, inputs, training=False):
        pixel_values, attention_mask = inputs
        batch_size = tf.shape(pixel_values)[0]
        
        # Patch embedding
        x = self.patch_embedding(pixel_values)  # (batch_size, max_patches, 2048) -> (batch_size, max_patches, 768)
        
        # Add position embeddings to the patch embeddings
        positions = tf.range(start=0, limit=self.max_patches, delta=1)  # Generate positional indices
        position_embeddings = self.position_embedding(positions)  # Position embedding (max_patches, 768)
        x += tf.expand_dims(position_embeddings, 0)  # Add position embeddings (batch_size, max_patches, 768)
        
        # Pass through transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training, mask=attention_mask)  # (batch_size, max_patches, 768)
        
        # Apply regression head to generate output
        x = self.regression_head(x)  # (batch_size, max_patches, 768) -> (batch_size, max_patches, 32768)
        
        # Reshape to match output size
        x = tf.reshape(x, (batch_size, self.max_patches, self.patch_size, self.patch_size, self.patch_shape[2]))  # (batch_size, max_patches, 16, 16, 128)
        
        # Apply ConvLSTM2D to capture temporal dynamics
        x = self.conv_lstm(x, training=training)  # (batch_size, 16, 16, 128)
        
        # Final convolution layer to refine the output
        x = self.conv_output(x)  # (batch_size, 16, 16, 128) -> (batch_size, 16, 16, 128)
        
        # Reshape for the final prediction
        x = tf.reshape(x, (batch_size, -1))  # (batch_size, 16, 16, 128) -> (batch_size, 32768)
        
        # Return the prediction for the first patch (T+1 time step)
        return x