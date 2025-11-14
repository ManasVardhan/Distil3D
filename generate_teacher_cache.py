"""
Generate Teacher Cache for Offline Distillation
Pre-compute Hunyuan3D predictions for all shoes
"""

import torch
from pathlib import Path
from PIL import Image
import trimesh
import numpy as np
from tqdm import tqdm
import json
import time

from test_hunyuan_teacher import HunyuanTeacher


class TeacherCacheGenerator:
    """Generate and cache teacher predictions for all shoes"""
    
    def __init__(self, config, cache_dir="teacher_cache", use_turbo=True):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Load teacher
        print("Loading Hunyuan3D teacher model...")
        self.teacher = HunyuanTeacher(use_turbo=use_turbo)
        
        # Metadata
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load or create metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            'generated': [],
            'failed': [],
            'stats': {}
        }
    
    def _save_metadata(self):
        """Save metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _is_cached(self, shoe_id):
        """Check if shoe already cached"""
        cache_file = self.cache_dir / f"{shoe_id}.pt"
        return cache_file.exists()
    
    def _prepare_images(self, shoe_id):
        """Load images for a shoe"""
        images_dir = Path(self.config.images_dir)
        
        images = {}
        for view in ['front', 'left', 'back']:
            img_path = images_dir / f"{shoe_id}_{view}.png"
            
            if not img_path.exists():
                # Try alternative naming
                img_path = images_dir / shoe_id / f"{view}.png"
            
            if not img_path.exists():
                raise FileNotFoundError(f"Missing {view} view for {shoe_id}")
            
            images[view] = Image.open(img_path).convert('RGB')
        
        return images
    
    def generate_for_shoe(self, shoe_id, num_steps=30):
        """Generate teacher prediction for one shoe"""
        
        try:
            # Load images
            images = self._prepare_images(shoe_id)
            
            # Generate prediction
            prediction = self.teacher.predict(images, num_steps=num_steps)
            
            # Save to cache
            cache_file = self.cache_dir / f"{shoe_id}.pt"
            torch.save(prediction, cache_file)
            
            # Update metadata
            self.metadata['generated'].append(shoe_id)
            self.metadata['stats'][shoe_id] = {
                'num_vertices': prediction['vertices'].shape[0],
                'num_faces': prediction['faces'].shape[0],
                'cache_size_mb': cache_file.stat().st_size / (1024 * 1024)
            }
            self._save_metadata()
            
            return True, None
            
        except Exception as e:
            error_msg = str(e)
            self.metadata['failed'].append({'shoe_id': shoe_id, 'error': error_msg})
            self._save_metadata()
            return False, error_msg
    
    def generate_all(self, num_steps=30, skip_existing=True):
        """Generate predictions for all shoes"""
        
        print("\n" + "="*70)
        print("GENERATING TEACHER CACHE FOR ALL SHOES")
        print("="*70)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Diffusion steps: {num_steps} (lower=faster, higher=better)")
        print(f"Skip existing: {skip_existing}")
        print("="*70 + "\n")
        
        # Get all shoes from dataset
        from load_data import ShoeDataset
        
        dataset = ShoeDataset(
            obj_dir=self.config.obj_dir,
            images_dir=self.config.images_dir,
            views=['front', 'back', 'left', 'right', 'top', 'bottom'],
            image_size=self.config.image_size
        )
        
        total_shoes = len(dataset.shoe_ids)
        print(f"Total shoes in dataset: {total_shoes}\n")
        
        # Filter already cached
        to_generate = []
        for shoe_id in dataset.shoe_ids:
            if skip_existing and self._is_cached(shoe_id):
                print(f"✓ Skipping {shoe_id} (already cached)")
            else:
                to_generate.append(shoe_id)
        
        print(f"\nShoes to generate: {len(to_generate)}")
        print(f"Already cached: {total_shoes - len(to_generate)}\n")
        
        if len(to_generate) == 0:
            print("All shoes already cached!")
            return
        
        # Estimate time
        time_per_shoe = 25 if not self.teacher else 10  # Turbo is ~10s
        estimated_minutes = (time_per_shoe * len(to_generate)) / 60
        print(f"Estimated time: {estimated_minutes:.1f} minutes")
        print("\nStarting generation...\n")
        
        # Generate
        start_time = time.time()
        success_count = 0
        
        for i, shoe_id in enumerate(tqdm(to_generate, desc="Generating")):
            shoe_start = time.time()
            
            success, error = self.generate_for_shoe(shoe_id, num_steps=num_steps)
            
            shoe_time = time.time() - shoe_start
            
            if success:
                success_count += 1
                tqdm.write(f"✓ {shoe_id}: {shoe_time:.1f}s")
            else:
                tqdm.write(f"✗ {shoe_id}: {error}")
            
            # Print progress every 10 shoes
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(to_generate) - i - 1)
                tqdm.write(f"\nProgress: {i+1}/{len(to_generate)} | "
                          f"Avg: {avg_time:.1f}s/shoe | "
                          f"ETA: {remaining/60:.1f} min\n")
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Success: {success_count}/{len(to_generate)}")
        print(f"Failed: {len(to_generate) - success_count}")
        print(f"Cached shoes: {len(self.metadata['generated'])}")
        print(f"Cache size: {sum(f.stat().st_size for f in self.cache_dir.glob('*.pt')) / (1024**2):.1f} MB")
        print("="*70)
        
        # Save final metadata
        self._save_metadata()
        
        print(f"\n✓ Teacher cache ready at: {self.cache_dir}")
        print("\nNext step: Train with distillation")
        print("  python train_with_distillation.py")


def main():
    """Main function"""
    from config import config
    
    print("\n" + "="*70)
    print("TEACHER CACHE GENERATION")
    print("="*70)
    
    # Configuration
    print("\nConfiguration:")
    print(f"  OBJ dir: {config.obj_dir}")
    print(f"  Images dir: {config.images_dir}")
    print(f"  Device: {config.device}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  WARNING: No GPU! Generation will be VERY slow")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Ask for quality vs speed
    print("\nQuality settings:")
    print("  1. Fast (10 steps, ~10s per shoe, ~17 min total)")
    print("  2. Balanced (20 steps, ~15s per shoe, ~25 min total)")
    print("  3. High quality (30 steps, ~25s per shoe, ~42 min total)")
    
    choice = input("\nChoose (1/2/3) [default: 1]: ").strip() or "1"
    
    steps_map = {'1': 10, '2': 20, '3': 30}
    num_steps = steps_map.get(choice, 10)
    
    print(f"\nUsing {num_steps} diffusion steps")
    
    # Generate cache
    generator = TeacherCacheGenerator(
        config=config,
        cache_dir="teacher_cache",
        use_turbo=True  # Always use turbo for speed
    )
    
    generator.generate_all(num_steps=num_steps, skip_existing=True)


if __name__ == "__main__":
    main()
