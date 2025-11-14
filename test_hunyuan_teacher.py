"""
Test Hunyuan3D Teacher Model
Verify it works with your shoe data
"""

import torch
from pathlib import Path
from PIL import Image
import trimesh
import numpy as np

# Try to import Hunyuan3D
try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    HUNYUAN_AVAILABLE = True
except ImportError:
    print("ERROR: Hunyuan3D not installed!")
    print("Run: bash install_hunyuan.sh")
    HUNYUAN_AVAILABLE = False
    exit(1)


class HunyuanTeacher:
    """Wrapper for Hunyuan3D teacher model"""
    
    def __init__(self, model_path='tencent/Hunyuan3D-2mv', use_turbo=True):
        print("Loading Hunyuan3D teacher model...")
        
        subfolder = 'hunyuan3d-dit-v2-mv-turbo' if use_turbo else 'hunyuan3d-dit-v2-mv'
        
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=True,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Teacher model loaded on {self.device}")
        print(f"  Mode: {'Turbo (fast)' if use_turbo else 'Standard (high quality)'}")
    
    def predict(self, images_dict, num_steps=30):
        """
        Generate 3D mesh from multiview images
        
        Args:
            images_dict: Dict with keys 'front', 'left', 'back'
                        Values are PIL Images or file paths
            num_steps: Diffusion steps (lower=faster, higher=better)
        
        Returns:
            dict with 'vertices', 'faces', 'normals'
        """
        
        # Prepare images
        prepared_images = {}
        for view in ['front', 'left', 'back']:
            if view not in images_dict:
                raise ValueError(f"Missing view: {view}")
            
            img = images_dict[view]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            
            prepared_images[view] = img
        
        print(f"Generating 3D mesh with {num_steps} steps...")
        
        # Run teacher model
        with torch.no_grad():
            mesh = self.pipeline(
                image=prepared_images,
                num_inference_steps=num_steps,
                octree_resolution=256,  # Lower for speed
                num_chunks=10000,
                generator=torch.manual_seed(42),
                output_type='trimesh'  # Get trimesh object
            )[0]
        
        print(f"✓ Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Extract data
        return {
            'vertices': torch.from_numpy(mesh.vertices.astype(np.float32)),
            'faces': torch.from_numpy(mesh.faces.astype(np.int64)),
            'vertex_normals': torch.from_numpy(mesh.vertex_normals.astype(np.float32))
        }


def test_on_sample_shoe(teacher, shoe_dir):
    """Test teacher on one shoe"""
    
    print("\n" + "="*70)
    print("TESTING ON SAMPLE SHOE")
    print("="*70)
    
    shoe_dir = Path(shoe_dir)
    
    # Load images
    images = {}
    for view in ['front', 'left', 'back']:
        # Try different naming conventions
        possible_names = [
            f"{view}.png",
            f"{shoe_dir.name}_{view}.png",
        ]
        
        found = False
        for name in possible_names:
            img_path = shoe_dir / name
            if img_path.exists():
                images[view] = img_path
                print(f"  Found {view}: {img_path}")
                found = True
                break
        
        if not found:
            print(f"  ERROR: Could not find {view} view image")
            return None
    
    # Generate prediction
    print("\nRunning teacher inference...")
    prediction = teacher.predict(images, num_steps=20)  # Fast test
    
    # Print results
    print("\nTeacher Output:")
    print(f"  Vertices: {prediction['vertices'].shape}")
    print(f"  Faces: {prediction['faces'].shape}")
    print(f"  Normals: {prediction['vertex_normals'].shape}")
    
    # Statistics
    verts = prediction['vertices']
    print(f"\nVertex Statistics:")
    print(f"  X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
    print(f"  Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
    print(f"  Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
    
    # Save for inspection
    output_path = Path("teacher_test_output.obj")
    mesh = trimesh.Trimesh(
        vertices=prediction['vertices'].numpy(),
        faces=prediction['faces'].numpy(),
        vertex_normals=prediction['vertex_normals'].numpy()
    )
    mesh.export(output_path)
    print(f"\n✓ Saved test output to: {output_path}")
    
    return prediction


def main():
    print("\n" + "="*70)
    print("HUNYUAN3D TEACHER MODEL TEST")
    print("="*70)
    
    if not HUNYUAN_AVAILABLE:
        return
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU found - will use CPU (very slow!)")
    
    # Load teacher
    teacher = HunyuanTeacher(use_turbo=True)  # Use turbo for speed
    
    # Test on sample
    print("\n" + "="*70)
    print("Please provide path to a shoe directory with images")
    print("Directory should contain: front.png, left.png, back.png")
    print("="*70)
    
    # Try to find a shoe automatically
    from config import config
    images_dir = Path(config.images_dir)
    
    # Find a shoe with all views
    shoe_found = False
    for shoe_id_file in Path(config.obj_dir).rglob("*.obj"):
        shoe_id = shoe_id_file.parent.name if shoe_id_file.parent != Path(config.obj_dir) else shoe_id_file.stem
        
        # Check if images exist
        has_views = all(
            (images_dir / f"{shoe_id}_{view}.png").exists()
            for view in ['front', 'left', 'back']
        )
        
        if has_views:
            print(f"\n✓ Found shoe with all views: {shoe_id}")
            
            # Create temp directory with correct naming
            temp_dir = Path(f"temp_test/{shoe_id}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for view in ['front', 'left', 'back']:
                src = images_dir / f"{shoe_id}_{view}.png"
                dst = temp_dir / f"{view}.png"
                Image.open(src).save(dst)
            
            result = test_on_sample_shoe(teacher, temp_dir)
            
            if result is not None:
                shoe_found = True
                print("\n" + "="*70)
                print("✓ SUCCESS! Teacher model works correctly")
                print("="*70)
                print("\nNext steps:")
                print("  1. Run: python generate_teacher_cache.py")
                print("  2. This will generate predictions for all 101 shoes")
                print("  3. Then train with: python train_with_distillation.py")
            
            break
    
    if not shoe_found:
        print("\n⚠ Could not find suitable test shoe")
        print("Please ensure your images directory contains:")
        print("  - {shoe_id}_front.png")
        print("  - {shoe_id}_left.png")
        print("  - {shoe_id}_back.png")


if __name__ == "__main__":
    main()
