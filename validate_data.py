"""
Data Validation Script - Simplified Version
Checks X (images) and Y (OBJ files) for correctness
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from collections import defaultdict
import json


class DataValidator:
    """Validates training data"""
    
    def __init__(self, obj_dir, images_dir):
        self.obj_dir = Path(obj_dir)
        self.images_dir = Path(images_dir)
        self.views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.issues = defaultdict(list)
        self.warnings = defaultdict(list)
        self.stats = {}
        
    def run_validation(self):
        """Run complete validation"""
        print("=" * 70)
        print("DATA VALIDATION")
        print("=" * 70)
        print(f"OBJ dir:    {self.obj_dir}")
        print(f"Images dir: {self.images_dir}")
        print("=" * 70)
        print()
        
        # Check directories
        if not self._check_directories():
            return False
        
        # Scan files
        obj_files = self._scan_obj_files()
        image_files = self._scan_image_files()
        
        # Validate mapping
        valid_pairs = self._validate_xy_mapping(obj_files, image_files)
        
        # Validate quality
        self._validate_obj_files(valid_pairs)
        self._validate_images(valid_pairs)
        self._check_data_quality(valid_pairs)
        
        # Print report
        self._print_report()
        
        return len(self.issues) == 0
    
    def _check_directories(self):
        """Check if directories exist"""
        print("TEST 1: Directory Existence")
        print("-" * 70)
        
        obj_ok = self.obj_dir.exists()
        img_ok = self.images_dir.exists()
        
        print(f"  OBJ directory:    {'EXISTS' if obj_ok else 'NOT FOUND'}")
        print(f"  Images directory: {'EXISTS' if img_ok else 'NOT FOUND'}")
        print()
        
        if not obj_ok:
            self.issues['directories'].append(f"OBJ directory not found: {self.obj_dir}")
        if not img_ok:
            self.issues['directories'].append(f"Images directory not found: {self.images_dir}")
        
        return obj_ok and img_ok
    
    def _scan_obj_files(self):
        """Scan for OBJ files"""
        print("TEST 2A: Scanning OBJ Files")
        print("-" * 70)
        
        obj_files = {}
        
        # Direct children
        for path in self.obj_dir.glob("*.obj"):
            obj_files[path.stem] = path
        
        # Subdirectories
        for subdir in self.obj_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            shoe_id = subdir.name
            objs = list(subdir.glob("*.obj"))
            
            if len(objs) == 1:
                obj_files[shoe_id] = objs[0]
            elif len(objs) > 1:
                obj_files[shoe_id] = objs[0]
                self.warnings['obj_files'].append(
                    f"Multiple OBJs in {shoe_id}, using {objs[0].name}"
                )
        
        print(f"  Found {len(obj_files)} OBJ files")
        if obj_files:
            print(f"  Example: {list(obj_files.keys())[0]}")
        else:
            self.issues['obj_files'].append("No OBJ files found")
        
        self.stats['total_objs'] = len(obj_files)
        print()
        return obj_files
    
    def _scan_image_files(self):
        """Scan for images"""
        print("TEST 2B: Scanning Images")
        print("-" * 70)
        
        image_files = defaultdict(dict)
        
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in self.images_dir.glob(ext):
                parts = img_path.stem.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                
                shoe_id, view = parts
                if view in self.views:
                    image_files[shoe_id][view] = img_path
        
        total = sum(len(v) for v in image_files.values())
        print(f"  Found {total} images for {len(image_files)} shoes")
        if image_files:
            example = list(image_files.keys())[0]
            print(f"  Example: {example} has {len(image_files[example])} views")
        else:
            self.issues['image_files'].append("No images found")
        
        self.stats['total_images'] = total
        print()
        return image_files
    
    def _validate_xy_mapping(self, obj_files, image_files):
        """Validate X-Y mapping"""
        print("TEST 3: X-Y Mapping")
        print("-" * 70)
        
        valid_pairs = {}
        incomplete = []
        
        for shoe_id, obj_path in obj_files.items():
            if shoe_id not in image_files:
                continue
            
            views = image_files[shoe_id]
            missing = [v for v in self.views if v not in views]
            
            if missing:
                incomplete.append((shoe_id, missing))
            else:
                valid_pairs[shoe_id] = {'obj': obj_path, 'images': views}
        
        print(f"  Valid pairs:     {len(valid_pairs)}")
        print(f"  Incomplete sets: {len(incomplete)}")
        
        if len(valid_pairs) == 0:
            self.issues['mapping'].append("No valid X-Y pairs found")
        
        if incomplete:
            self.warnings['mapping'].append(f"{len(incomplete)} shoes missing some views")
        
        self.stats['valid_pairs'] = len(valid_pairs)
        print()
        return valid_pairs
    
    def _validate_obj_files(self, valid_pairs):
        """Validate OBJ quality"""
        print("TEST 4: OBJ Validation")
        print("-" * 70)
        
        if not valid_pairs:
            print("  Skipping (no valid pairs)")
            print()
            return
        
        stats = {'valid': 0, 'corrupt': 0, 'empty': 0}
        sample = list(valid_pairs.keys())[:5]
        
        print(f"  Validating {len(sample)} samples...")
        
        for shoe_id in sample:
            try:
                mesh = trimesh.load(valid_pairs[shoe_id]['obj'], process=False)
                if len(mesh.vertices) == 0:
                    stats['empty'] += 1
                else:
                    stats['valid'] += 1
            except:
                stats['corrupt'] += 1
        
        print(f"    Valid:   {stats['valid']}/{len(sample)}")
        print(f"    Corrupt: {stats['corrupt']}")
        print(f"    Empty:   {stats['empty']}")
        print()
    
    def _validate_images(self, valid_pairs):
        """Validate images"""
        print("TEST 5: Image Validation")
        print("-" * 70)
        
        if not valid_pairs:
            print("  Skipping (no valid pairs)")
            print()
            return
        
        stats = {'valid': 0, 'corrupt': 0, 'resolutions': set()}
        sample = list(valid_pairs.keys())[:5]
        
        print(f"  Validating {len(sample)} samples...")
        
        for shoe_id in sample:
            for view, img_path in valid_pairs[shoe_id]['images'].items():
                try:
                    img = Image.open(img_path)
                    stats['resolutions'].add(img.size)
                    stats['valid'] += 1
                except:
                    stats['corrupt'] += 1
        
        print(f"    Valid:       {stats['valid']}")
        print(f"    Corrupt:     {stats['corrupt']}")
        print(f"    Resolutions: {stats['resolutions']}")
        print()
    
    def _check_data_quality(self, valid_pairs):
        """Check data quality"""
        print("TEST 6: Data Quality")
        print("-" * 70)
        
        num_pairs = len(valid_pairs)
        
        if num_pairs < 10:
            self.warnings['quality'].append(
                f"Only {num_pairs} pairs - need at least 50"
            )
            print(f"  WARNING: Only {num_pairs} pairs (need 50+)")
        elif num_pairs < 50:
            self.warnings['quality'].append(
                f"{num_pairs} pairs - more would be better"
            )
            print(f"  OK: {num_pairs} pairs (50+ recommended)")
        else:
            print(f"  GOOD: {num_pairs} pairs")
        
        print()
    
    def _print_report(self):
        """Print report"""
        print("=" * 70)
        print("VALIDATION REPORT")
        print("=" * 70)
        
        print("\nStatistics:")
        print(f"  Total OBJs:      {self.stats.get('total_objs', 0)}")
        print(f"  Total images:    {self.stats.get('total_images', 0)}")
        print(f"  Valid pairs:     {self.stats.get('valid_pairs', 0)}")
        
        if self.issues:
            print("\nCRITICAL ISSUES:")
            for category, issues in self.issues.items():
                print(f"\n  {category}:")
                for issue in issues:
                    print(f"    - {issue}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for category, warnings in self.warnings.items():
                print(f"\n  {category}:")
                for warning in warnings[:3]:
                    print(f"    - {warning}")
        
        print("\n" + "=" * 70)
        if not self.issues:
            print("PASSED - Data is ready")
        else:
            print("FAILED - Fix issues above")
        print("=" * 70)
        
        # Save report
        report = {
            'statistics': self.stats,
            'issues': dict(self.issues),
            'warnings': dict(self.warnings),
            'passed': len(self.issues) == 0
        }
        
        with open('validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nReport saved to: validation_report.json")


def main():
    """Main function"""
    
    # Load config
    try:
        from config import config
        obj_dir = config.obj_dir
        images_dir = config.images_dir
        print("Loaded paths from config.py\n")
    except:
        obj_dir = input("OBJ directory: ").strip()
        images_dir = input("Images directory: ").strip()
        print()
    
    # Run validation
    validator = DataValidator(obj_dir, images_dir)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()