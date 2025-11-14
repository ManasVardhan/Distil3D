"""
MEMORY-OPTIMIZED Configuration for Mac MPS
Reduced settings to fit in 9GB memory
"""

import torch
from pathlib import Path

class FixedConfig:
    """Configuration optimized for Mac MPS memory constraints"""
    
    # ========================================================================
    # Data Paths
    # ========================================================================
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/OBJs"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    # Output directories
    checkpoint_dir = "checkpoints_fixed"
    log_dir = "logs_fixed"
    output_dir = "output_fixed"
    
    # ========================================================================
    # Model Architecture Settings
    # ========================================================================
    num_points = 2562  # Final mesh vertices
    
    hidden_dim = 768  # REDUCED from 1024 to save memory
    feature_dim = 768  # DINOv2 base
    num_views = 6
    
    # ========================================================================
    # Stage 1: Geometry Training Settings  
    # ========================================================================
    batch_size_stage1 = 1
    num_epochs_stage1 = 150
    
    # CRITICAL: Very low learning rate
    learning_rate_stage1 = 1e-5
    
    weight_decay_stage1 = 0.01
    freeze_image_encoder = True  # CHANGED: Freeze to save memory during training
    
    # ========================================================================
    # Loss Function Weights
    # ========================================================================
    lambda_chamfer = 1.0
    lambda_edge = 2.0      # Strong
    lambda_smooth = 1.0    # Strong
    lambda_normal = 0.5
    
    # ========================================================================
    # Stage 2: Texture Training Settings
    # ========================================================================
    batch_size_stage2 = 1
    num_epochs_stage2 = 30
    learning_rate_stage2 = 1e-4
    weight_decay_stage2 = 0.01
    
    lambda_color_l1l2 = 1.0
    lambda_perceptual = 0.5
    lambda_smooth_texture = 0.1
    
    # ========================================================================
    # Data Loading Settings - MEMORY OPTIMIZATION
    # ========================================================================
    image_size = 384  # REDUCED from 512 to save memory (saves ~40% memory)
    num_workers = 0   # Mac MPS requires 0
    
    # ========================================================================
    # Device Settings
    # ========================================================================
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========================================================================
    # Logging & Checkpointing
    # ========================================================================
    log_interval = 5
    save_interval = 10
    
    # ========================================================================
    # Early Stopping
    # ========================================================================
    patience = 25
    min_delta = 1e-5
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    geometry_checkpoint = "checkpoints_fixed/multiscale_best.pth"
    texture_checkpoint = "checkpoints/texture_best.pth"
    
    mesh_extraction_method = 'direct'
    inference_resolution = 384  # Match training resolution
    
    # ========================================================================
    # Advanced Settings
    # ========================================================================
    grad_clip_norm = 0.5
    use_amp = False  # MPS doesn't support AMP yet
    seed = 42
    val_split = 0.1
    
    # Warmup settings
    warmup_epochs = 10
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("MEMORY-OPTIMIZED MULTI-SCALE MESH TRAINING")
        print("="*70)
        print(f"Data:")
        print(f"  OBJ dir:    {cls.obj_dir}")
        print(f"  Images dir: {cls.images_dir}")
        print(f"\nMemory Optimizations:")
        print(f"  ✓ Image size: {cls.image_size} (reduced from 512)")
        print(f"  ✓ Hidden dim: {cls.hidden_dim} (reduced from 1024)")
        print(f"  ✓ Encoder frozen: {cls.freeze_image_encoder} (saves gradients)")
        print(f"\nAnti-Fragmentation Settings:")
        print(f"  ✓ Learning rate: {cls.learning_rate_stage1}")
        print(f"  ✓ Edge loss: {cls.lambda_edge}")
        print(f"  ✓ Smooth loss: {cls.lambda_smooth}")
        print(f"  ✓ Gradient clip: {cls.grad_clip_norm}")
        print(f"\nStage 1 (Geometry):")
        print(f"  Epochs:       {cls.num_epochs_stage1}")
        print(f"  Batch size:   {cls.batch_size_stage1}")
        print(f"  Final verts:  {cls.num_points}")
        print(f"\nDevice: {cls.device}")
        print("="*70)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.log_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.output_dir).mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def get_training_summary(cls):
        """Get a summary of key training parameters"""
        return {
            'learning_rate': cls.learning_rate_stage1,
            'epochs': cls.num_epochs_stage1,
            'regularization': {
                'edge': cls.lambda_edge,
                'smooth': cls.lambda_smooth,
                'normal': cls.lambda_normal
            },
            'grad_clip': cls.grad_clip_norm,
            'warmup_epochs': cls.warmup_epochs,
            'memory_optimized': True
        }


# Create singleton config instance
config = FixedConfig()

# Set random seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

# Create directories on import
config.create_directories()


# ============================================================================
# MEMORY OPTIMIZATION NOTES
# ============================================================================
"""
Changes for Mac MPS (9GB limit):

1. image_size: 512 → 384
   - Saves ~40% memory in image encoder
   - Still sufficient for 3D reconstruction

2. hidden_dim: 1024 → 768
   - Reduces decoder memory footprint
   - Matches DINOv2 feature dim (cleaner)

3. freeze_image_encoder: False → True
   - Saves gradient memory during training
   - Encoder is pre-trained anyway
   - Can fine-tune later if needed

These changes reduce peak memory from ~9GB to ~6GB while maintaining quality.

If still OOM, try:
- image_size = 256 (saves 60% memory, may reduce quality)
- Use CPU for inference only
- Close other applications
"""