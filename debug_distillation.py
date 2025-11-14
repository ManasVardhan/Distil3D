"""
Debug script to identify the source of CUDA index errors
"""

import torch
import os
import sys
from pathlib import Path

# Set CUDA environment for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("="*70)
print("DISTILLATION DEBUGGING SCRIPT")
print("="*70)

# Import modules
print("\n1. Importing modules...")
try:
    from train_with_distillation import (
        DistillationDataset, 
        DistillationLoss,
        distillation_collate_fn,
        chamfer_distance,
        sample_points_from_mesh
    )
    from config import config
    from models.geometry_model import ImprovedGeometryModel
    from torch.utils.data import DataLoader
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Check teacher cache
print("\n2. Checking teacher cache...")
cache_dir = Path("teacher_cache")
if not cache_dir.exists():
    print(f"✗ Teacher cache not found at {cache_dir}")
    sys.exit(1)

cache_files = list(cache_dir.glob("*.pt"))
print(f"✓ Found {len(cache_files)} cached files")

# Load dataset
print("\n3. Loading dataset...")
try:
    dataset = DistillationDataset(
        obj_dir=config.obj_dir,
        images_dir=config.images_dir,
        views=['front', 'back', 'left', 'right', 'top', 'bottom'],
        image_size=config.image_size,
        teacher_cache_dir="teacher_cache"
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"✗ Dataset loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test single sample
print("\n4. Testing single sample...")
try:
    sample = dataset[0]
    print(f"✓ Sample loaded: {sample['shoe_id']}")
    print(f"  Vertices shape: {sample['vertices'].shape}")
    print(f"  Faces shape: {sample['faces'].shape}")
    
    if sample['teacher_mesh'] is not None:
        print(f"  Teacher vertices shape: {sample['teacher_mesh']['vertices'].shape}")
        if sample['teacher_mesh']['faces'] is not None:
            print(f"  Teacher faces shape: {sample['teacher_mesh']['faces'].shape}")
            
            # Check for invalid indices
            max_idx = sample['teacher_mesh']['faces'].max().item()
            num_verts = len(sample['teacher_mesh']['vertices'])
            print(f"  Teacher face max index: {max_idx}")
            print(f"  Teacher num vertices: {num_verts}")
            
            if max_idx >= num_verts:
                print(f"  ⚠ WARNING: Invalid face indices! Max {max_idx} >= {num_verts} verts")
            else:
                print(f"  ✓ Face indices valid")
        else:
            print(f"  Teacher faces: None")
    else:
        print(f"  ✗ No teacher mesh!")
except Exception as e:
    print(f"✗ Sample loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataloader
print("\n5. Testing dataloader...")
try:
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=distillation_collate_fn,
        num_workers=0  # Single-threaded for debugging
    )
    
    batch = next(iter(dataloader))
    print(f"✓ Batch loaded")
    print(f"  Batch size: {len(batch['vertices'])}")
    print(f"  Images: {list(batch['images'].keys())}")
    print(f"  Teacher meshes: {len(batch['teacher_mesh'])}")
    
    # Check each teacher mesh
    for i, tm in enumerate(batch['teacher_mesh']):
        if tm is not None:
            print(f"\n  Sample {i}:")
            print(f"    Teacher verts: {tm['vertices'].shape}")
            if tm['faces'] is not None:
                print(f"    Teacher faces: {tm['faces'].shape}")
                max_idx = tm['faces'].max().item()
                num_verts = len(tm['vertices'])
                print(f"    Face max index: {max_idx}, Num verts: {num_verts}")
                if max_idx >= num_verts:
                    print(f"    ⚠ INVALID FACES!")
        else:
            print(f"  Sample {i}: No teacher mesh")
            
except Exception as e:
    print(f"✗ Dataloader error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model
print("\n6. Testing model...")
device = config.device
try:
    # Use config attributes if available, otherwise use defaults
    freeze_encoder = getattr(config, 'freeze_image_encoder', True)
    hidden_dim = getattr(config, 'hidden_dim', 256)
    
    model = ImprovedGeometryModel(
        freeze_encoder=freeze_encoder,
        hidden_dim=hidden_dim
    ).to(device)
    print(f"✓ Model created on {device}")
    
    # Test forward pass
    images_single = {k: v[0:1].to(device) for k, v in batch['images'].items()}
    with torch.no_grad():
        output = model(images_single, return_all_levels=True)
    print(f"✓ Forward pass successful")
    print(f"  Output vertices shape: {output['fine'][0].shape}")
    print(f"  Output faces shape: {output['faces_fine'].shape}")
    
except Exception as e:
    print(f"✗ Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test chamfer distance
print("\n7. Testing chamfer distance...")
try:
    # Create simple test tensors
    pred = torch.randn(100, 3).to(device)
    target = torch.randn(150, 3).to(device)
    
    dist = chamfer_distance(pred, target)
    print(f"✓ Chamfer distance computed: {dist.item():.6f}")
    
    # Test with different shapes
    pred_batch = torch.randn(2, 100, 3).to(device)
    target_single = torch.randn(150, 3).to(device)
    dist = chamfer_distance(pred_batch, target_single)
    print(f"✓ Batch chamfer distance computed: {dist.item():.6f}")
    
except Exception as e:
    print(f"✗ Chamfer distance error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loss computation
print("\n8. Testing loss computation...")
try:
    # Use config attributes if available, otherwise use defaults
    lambda_chamfer = getattr(config, 'lambda_chamfer', 1.0)
    lambda_edge = getattr(config, 'lambda_edge', 2.0)
    
    criterion = DistillationLoss(
        alpha=0.9,
        lambda_chamfer=lambda_chamfer,
        lambda_edge=lambda_edge
    )
    
    # Get first sample with valid teacher
    sample_idx = 0
    for i, tm in enumerate(batch['teacher_mesh']):
        if tm is not None:
            sample_idx = i
            break
    
    gt_vertices = batch['vertices'][sample_idx].to(device)
    teacher_mesh = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch['teacher_mesh'][sample_idx].items()
    }
    
    print(f"\n  Using sample {sample_idx}:")
    print(f"    GT vertices: {gt_vertices.shape}")
    print(f"    Teacher vertices: {teacher_mesh['vertices'].shape}")
    print(f"    Student vertices: {output['fine'][0].shape}")
    
    # Check student faces
    if 'faces_fine' in output and output['faces_fine'] is not None:
        print(f"    Student faces: {output['faces_fine'].shape}")
        if len(output['faces_fine']) > 0:
            print(f"    Student faces range: [{output['faces_fine'].min().item()}, {output['faces_fine'].max().item()}]")
    else:
        print(f"    Student faces: None")
    
    # Compute loss
    loss, loss_dict = criterion(
        output,
        teacher_mesh,
        gt_vertices,
        output['faces_fine']
    )
    
    print(f"\n✓ Loss computation successful!")
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Chamfer: {loss_dict['chamfer']:.6f}")
    print(f"  Teacher: {loss_dict['teacher']:.6f}")
    print(f"  GT: {loss_dict['gt']:.6f}")
    print(f"  Edge: {loss_dict['edge']:.6f}")
    
except Exception as e:
    print(f"\n✗ Loss computation error: {e}")
    import traceback
    traceback.print_exc()
    
    # Additional debugging
    print("\nAdditional debugging info:")
    print(f"  device: {device}")
    if 'gt_vertices' in locals():
        print(f"  gt_vertices shape: {gt_vertices.shape}, device: {gt_vertices.device}")
    if 'teacher_mesh' in locals():
        print(f"  teacher_mesh vertices shape: {teacher_mesh['vertices'].shape}, device: {teacher_mesh['vertices'].device}")
        if teacher_mesh['faces'] is not None:
            print(f"  teacher_mesh faces shape: {teacher_mesh['faces'].shape}, device: {teacher_mesh['faces'].device}")
            print(f"  teacher_mesh faces range: [{teacher_mesh['faces'].min().item()}, {teacher_mesh['faces'].max().item()}]")
    if 'output' in locals():
        print(f"  output['fine'][0] shape: {output['fine'][0].shape}, device: {output['fine'][0].device}")
    
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nThe distillation pipeline is working correctly.")
print("You can now run: python train_with_distillation.py")