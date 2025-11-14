"""
UPDATED: Dataset Loader for Multi-View to 3D Mesh Training
Now handles OBJ files in subdirectories
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import trimesh


# ============================================================================
# OBJ to Training Target Converter (unchanged)
# ============================================================================

class OBJToTrainingTarget:
    """Convert 3D OBJ files to training targets (Y values)"""
    
    def __init__(self, normalize=True, max_vertices=10000):
        self.normalize = normalize
        self.max_vertices = max_vertices
    
    def load_and_process(self, obj_path):
        """Load OBJ and convert to training target format"""
        print(f"Loading: {obj_path}")
        
        # Load mesh
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
        # Extract geometry
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        print(f"  Original: {len(vertices)} vertices, {len(faces)} faces")
        
        # Optional simplification if too many vertices
        if len(vertices) > self.max_vertices:
            print(f"  Simplifying mesh to ≤ {self.max_vertices} vertices...")
            target_faces = int((self.max_vertices / len(vertices)) * len(faces))
            target_faces = max(100, target_faces)  # Ensure at least 100 faces
            
            try:
                # Try quadric decimation first
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if len(simplified.vertices) < len(vertices):
                    mesh = simplified
                    vertices = mesh.vertices.copy()
                    faces = mesh.faces.copy()
                    print(f"  Simplified: {len(vertices)} vertices, {len(faces)} faces")
                else:
                    # Fall back to mesh-aware vertex sampling
                    print(f"  Simplification didn't reduce vertices, using mesh-aware sampling...")
                    
                    # Build vertex-to-faces adjacency for mesh-aware sampling
                    from collections import defaultdict
                    vertex_faces = defaultdict(list)
                    for face_idx, face in enumerate(faces):
                        for v in face:
                            vertex_faces[v].append(face_idx)
                    
                    # Grow selection from seeds to keep mesh connectivity
                    selected = set()
                    candidates = list(range(len(vertices)))
                    np.random.shuffle(candidates)
                    
                    for seed in candidates:
                        if len(selected) >= self.max_vertices:
                            break
                        if seed not in selected:
                            selected.add(seed)
                            # Add face neighbors to increase face retention
                            for face_idx in vertex_faces[seed][:3]:  # Limit to 3 faces per vertex
                                face = faces[face_idx]
                                for v in face:
                                    if len(selected) < self.max_vertices:
                                        selected.add(v)
                    
                    # Convert to sorted array
                    indices = np.array(sorted(list(selected)))[:self.max_vertices]
                    
                    # Create mapping from old to new indices
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                    
                    # Sample vertices
                    vertices = vertices[indices]
                    
                    # Rebuild faces - only keep valid faces
                    new_faces = []
                    for face in faces:
                        if all(v in old_to_new for v in face):
                            new_face = [old_to_new[v] for v in face]
                            new_faces.append(new_face)
                    
                    faces = np.array(new_faces) if len(new_faces) > 0 else np.array([[0, 1, 2]])
                    print(f"  Sampled: {len(vertices)} vertices, {len(faces)} faces")
            except Exception as e:
                print(f"  Warning: Simplification failed ({e})")
                # Force reduction by mesh-aware sampling
                if len(vertices) > self.max_vertices:
                    print(f"  Forcing mesh-aware vertex sampling to {self.max_vertices}...")
                    
                    # Build vertex-to-faces adjacency for connectivity-preserving sampling
                    from collections import defaultdict
                    vertex_faces = defaultdict(list)
                    for face_idx, face in enumerate(faces):
                        for v in face:
                            vertex_faces[v].append(face_idx)
                    
                    # Grow selection from random seeds, prioritizing mesh connectivity
                    selected = set()
                    candidates = list(range(len(vertices)))
                    np.random.shuffle(candidates)
                    
                    for seed in candidates:
                        if len(selected) >= self.max_vertices:
                            break
                        if seed not in selected:
                            selected.add(seed)
                            # Add neighbors from connected faces to preserve topology
                            for face_idx in vertex_faces[seed][:3]:  # Limit neighbors per vertex
                                if len(selected) >= self.max_vertices:
                                    break
                                face = faces[face_idx]
                                for v in face:
                                    if len(selected) < self.max_vertices:
                                        selected.add(v)
                                    else:
                                        break
                    
                    # Convert to sorted array for better memory locality
                    indices = np.array(sorted(list(selected)))[:self.max_vertices]
                    
                    # Create mapping from old to new indices
                    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
                    
                    # Sample vertices
                    vertices = vertices[indices]
                    
                    # Rebuild faces - ONLY keep faces where ALL vertices are in sampled set
                    new_faces = []
                    for face in faces:
                        # Check if all vertices in this face are in our sampled set
                        if all(v in old_to_new for v in face):
                            # Remap to new indices
                            new_face = [old_to_new[v] for v in face]
                            new_faces.append(new_face)
                    
                    # Convert to numpy, ensure we have at least one face
                    if len(new_faces) > 0:
                        faces = np.array(new_faces)
                    else:
                        # Fallback: create a simple triangle from first 3 vertices
                        faces = np.array([[0, 1, 2]])
                    
                    print(f"  Forced: {len(vertices)} vertices, {len(faces)} faces")

        
        # Normalize mesh
        if self.normalize:
            vertices = self._normalize_vertices(vertices)
            print(f"  Normalized to unit sphere")
        
        # Extract vertex colors
        vertex_colors = self._extract_vertex_colors(mesh, len(vertices))
        
        # Compute vertex normals
        vertex_normals = self._compute_vertex_normals(mesh, vertices, faces)
        
        # Pack into dictionary
        y_values = {
            'vertices': vertices.astype(np.float32),
            'faces': faces.astype(np.int64),
            'vertex_colors': vertex_colors,
            'vertex_normals': vertex_normals.astype(np.float32)
        }
        
        print(f"  ✓ Converted to training target")
        
        return y_values
    
    def _normalize_vertices(self, vertices):
        """Center at origin and normalize to unit sphere"""
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        max_dist = np.linalg.norm(vertices, axis=1).max()
        if max_dist > 0:
            vertices = vertices / max_dist
        return vertices
    
    def _extract_vertex_colors(self, mesh, num_vertices):
        """Extract vertex colors from mesh"""
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            print(f"  Found vertex colors")
            return colors
        
        if hasattr(mesh.visual, 'uv') and hasattr(mesh.visual, 'material'):
            try:
                colors = self._sample_texture_at_vertices(mesh)
                if colors is not None:
                    print(f"  Extracted colors from texture")
                    return colors
            except Exception as e:
                print(f"  Could not extract texture colors: {e}")
        
        if hasattr(mesh.visual, 'material'):
            try:
                material = mesh.visual.material
                if hasattr(material, 'diffuse'):
                    color = material.diffuse[:3] / 255.0
                    colors = np.tile(color, (num_vertices, 1)).astype(np.float32)
                    print(f"  Using material diffuse color: {color}")
                    return colors
            except:
                pass
        
        print(f"  No colors found, using white")
        return np.ones((num_vertices, 3), dtype=np.float32)
    
    def _sample_texture_at_vertices(self, mesh):
        """Sample texture at vertex UV coordinates"""
        if not hasattr(mesh.visual, 'material'):
            return None
        if not hasattr(mesh.visual.material, 'image'):
            return None
        
        texture_img = np.array(mesh.visual.material.image).astype(np.float32) / 255.0
        h, w = texture_img.shape[:2]
        uv = mesh.visual.uv
        
        u_coords = (uv[:, 0] * (w - 1)).astype(np.int32)
        v_coords = ((1 - uv[:, 1]) * (h - 1)).astype(np.int32)
        u_coords = np.clip(u_coords, 0, w - 1)
        v_coords = np.clip(v_coords, 0, h - 1)
        
        colors = texture_img[v_coords, u_coords, :3]
        return colors.astype(np.float32)
    
    def _compute_vertex_normals(self, mesh, vertices, faces):
        """Compute smooth vertex normals"""
        try:
            normals = mesh.vertex_normals.copy()
        except:
            normals = self._compute_normals_manual(vertices, faces)
        return normals.astype(np.float32)
    
    def _compute_normals_manual(self, vertices, faces):
        """Manually compute vertex normals"""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 0:
                face_normal = face_normal / norm
            for idx in face:
                normals[idx] += face_normal
        
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norms > 0
        normals[valid] /= norms[valid]
        return normals


# ============================================================================
# UPDATED Dataset Class - Handles Subdirectories
# ============================================================================

class MultiViewMeshDataset(Dataset):
    """
    UPDATED: Dataset that loads from subdirectories
    - X: six-view rendered images
    - Y: 3D mesh (vertices, faces, colors, normals)
    """

    def __init__(self, obj_dir, images_dir, views, image_size=512):
        super().__init__()
        self.obj_dir = Path(obj_dir)
        self.images_dir = Path(images_dir)
        self.views = views
        self.image_size = image_size
        
        # Create converter
        self.obj_converter = OBJToTrainingTarget(normalize=True, max_vertices=10000)
        
        # Build mapping: now handles subdirectories
        self.obj_paths, self.shoe_ids = self._scan_objs()
        
        # Verify X-Y mapping
        self._verify_xy_mapping()

    def _scan_objs(self):
        """
        UPDATED: Scan OBJ directory including subdirectories
        Looks for .obj files in:
        1. Direct children: OBJs/shoe001.obj
        2. Subdirectories: OBJs/shoe001/shoe001.obj
        """
        print("\nScanning OBJ directory (including subdirectories)...")
        obj_paths = []
        shoe_ids = []
        
        # Method 1: Check direct children
        direct_objs = list(self.obj_dir.glob("*.obj"))
        for path in direct_objs:
            shoe_id = path.stem
            obj_paths.append(path)
            shoe_ids.append(shoe_id)
        
        # Method 2: Check subdirectories (shoe_id/shoe_id.obj pattern)
        subdirs = [d for d in self.obj_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            shoe_id = subdir.name
            
            # Look for .obj files in subdirectory
            obj_files = list(subdir.glob("*.obj"))
            
            if len(obj_files) == 0:
                continue  # No OBJ in this subdirectory
            elif len(obj_files) == 1:
                # Single OBJ file - use it
                obj_paths.append(obj_files[0])
                shoe_ids.append(shoe_id)
            else:
                # Multiple OBJ files - prefer one matching shoe_id
                matching = [f for f in obj_files if f.stem == shoe_id]
                if matching:
                    obj_paths.append(matching[0])
                    shoe_ids.append(shoe_id)
                else:
                    # Just use the first one
                    obj_paths.append(obj_files[0])
                    shoe_ids.append(shoe_id)
                    print(f"  Warning: Multiple OBJs in {subdir.name}, using {obj_files[0].name}")
        
        # Sort for consistency
        sorted_pairs = sorted(zip(shoe_ids, obj_paths))
        shoe_ids = [pair[0] for pair in sorted_pairs]
        obj_paths = [pair[1] for pair in sorted_pairs]
        
        print(f"  Found {len(obj_paths)} OBJ files")
        if len(obj_paths) > 0:
            print(f"  Example: {obj_paths[0]}")
        
        return obj_paths, shoe_ids

    def _verify_xy_mapping(self):
        """Ensure each OBJ has corresponding multi-view images"""
        print("\nVerifying X-Y mappings (OBJ ↔ images)...")
        print("="*60)
        
        missing_images = []
        valid_pairs = 0
        
        for idx, (obj_path, shoe_id) in enumerate(zip(self.obj_paths, self.shoe_ids)):
            missing_views = []
            
            for view in self.views:
                img_path_png = self.images_dir / f"{shoe_id}_{view}.png"
                img_path_jpeg = self.images_dir / f"{shoe_id}_{view}.jpeg"
                img_path_jpg = self.images_dir / f"{shoe_id}_{view}.jpg"
                
                if not (img_path_png.exists() or img_path_jpeg.exists() or img_path_jpg.exists()):
                    missing_views.append(view)
            
            if missing_views:
                missing_images.append((shoe_id, missing_views))
            else:
                valid_pairs += 1
        
        print(f"  Valid pairs:   {valid_pairs}")
        print(f"  Missing pairs: {len(missing_images)}")
        
        if missing_images:
            print("\nMissing views per shoe:")
            for shoe_id, views in missing_images[:5]:  # Show first 5
                print(f"  {shoe_id}: missing {views}")
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
            
            # Filter out invalid pairs
            valid_indices = []
            for i, (shoe_id, _) in enumerate(zip(self.shoe_ids, self.obj_paths)):
                if not any(shoe_id == m_id for m_id, _ in missing_images):
                    valid_indices.append(i)
            
            self.obj_paths = [self.obj_paths[i] for i in valid_indices]
            self.shoe_ids = [self.shoe_ids[i] for i in valid_indices]
            
            print(f"\n✓ Dataset cleaned: {len(self.obj_paths)} valid pairs remaining")
        else:
            print("✓ All X-Y mappings verified successfully!")
        
        print("="*60 + "\n")
    
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        shoe_id = self.shoe_ids[idx]
        
        # Load Y values (ground truth mesh)
        y_values = self.obj_converter.load_and_process(obj_path)
        
        # Load X values (multi-view images)
        images = {}
        for view in self.views:
            img_path_png = self.images_dir / f"{shoe_id}_{view}.png"
            img_path_jpeg = self.images_dir / f"{shoe_id}_{view}.jpeg"
            img_path_jpg = self.images_dir / f"{shoe_id}_{view}.jpg"
            
            if img_path_png.exists():
                img_path = img_path_png
            elif img_path_jpeg.exists():
                img_path = img_path_jpeg
            elif img_path_jpg.exists():
                img_path = img_path_jpg
            else:
                raise FileNotFoundError(f"No image found for {shoe_id} view {view}")
            
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images[view] = img_tensor
        
        sample = {
            'images': images,
            'vertices': torch.from_numpy(y_values['vertices']),
            'faces': torch.from_numpy(y_values['faces']),
            'vertex_colors': torch.from_numpy(y_values['vertex_colors']),
            'vertex_normals': torch.from_numpy(y_values['vertex_normals']),
            'shoe_id': shoe_id,
            'obj_path': str(obj_path)
        }
        
        return sample


# ============================================================================
# Collate Function (unchanged)
# ============================================================================

def collate_fn(batch):
    """Custom collate function to handle dictionaries"""
    views = batch[0]['images'].keys()
    images_batch = {}
    for view in views:
        images_batch[view] = torch.stack([item['images'][view] for item in batch], dim=0)
    
    vertices_batch = [item['vertices'] for item in batch]
    faces_batch = [item['faces'] for item in batch]
    colors_batch = [item['vertex_colors'] for item in batch]
    normals_batch = [item['vertex_normals'] for item in batch]
    shoe_ids = [item['shoe_id'] for item in batch]
    obj_paths = [item['obj_path'] for item in batch]

    return {
        'images': images_batch,
        'vertices': vertices_batch,
        'faces': faces_batch,
        'vertex_colors': colors_batch,
        'vertex_normals': normals_batch,
        'shoe_id': shoe_ids,
        'obj_path': obj_paths
    }


# ============================================================================
# Aliases for backward compatibility
# ============================================================================

ShoeDataset = MultiViewMeshDataset
custom_collate_fn = collate_fn