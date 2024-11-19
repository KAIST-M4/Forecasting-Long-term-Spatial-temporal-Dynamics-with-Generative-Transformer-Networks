class ViTConfig:
    def __init__(self, image_size=8, patch_size=8, patch_shape=(8, 8, 32), num_layers=12, hidden_size=768, num_heads=12, mlp_dim=3072, max_patches=5):
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_shape = patch_shape
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.max_patches = max_patches







