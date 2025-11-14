"""
COMPREHENSIVE FIX for Mesh Fragmentation
This addresses the fundamental architectural issues
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor
import trimesh


class MultiViewImageEncoder(nn.Module):
    """Encode images using DINOv2"""
    
    def __init__(self, model_name='facebook/dinov2-base', freeze=False):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        outputs = self.encoder(pixel_values=images)
        features = outputs.last_hidden_state[:, 0]
        return features


class ViewAggregator(nn.Module):
    """Aggregate features from 6 views with attention"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, view_features):
        # view_features: (B, 6, feature_dim)
        
        # Attention weights
        attention_scores = self.attention(view_features)  # (B, 6, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        weighted = torch.sum(view_features * attention_weights, dim=1)
        
        # Max pooling
        max_pool = torch.max(view_features, dim=1)[0]
        
        # Fusion
        aggregated = torch.cat([weighted, max_pool], dim=1)
        output = self.fusion(aggregated)
        return output


class CoarseToFineDecoder(nn.Module):
    """
    FIXED: Hierarchical mesh decoder that progressively refines
    Key improvements:
    1. Multi-scale processing (coarse -> fine)
    2. Explicit mesh connectivity preservation
    3. Bounded deformations at each level
    """
    
    def __init__(self, feature_dim=768, hidden_dim=1024):
        super().__init__()
        
        # Create multi-resolution mesh templates
        # Level 0: 162 vertices (subdivision=2)
        sphere_coarse = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        self.register_buffer('vertices_coarse', 
                           torch.tensor(sphere_coarse.vertices, dtype=torch.float32))
        self.register_buffer('faces_coarse',
                           torch.tensor(sphere_coarse.faces, dtype=torch.long))
        
        # Level 1: 642 vertices (subdivision=3)
        sphere_mid = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        self.register_buffer('vertices_mid',
                           torch.tensor(sphere_mid.vertices, dtype=torch.float32))
        self.register_buffer('faces_mid',
                           torch.tensor(sphere_mid.faces, dtype=torch.long))
        
        # Level 2: 2562 vertices (subdivision=4)
        sphere_fine = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        self.register_buffer('vertices_fine',
                           torch.tensor(sphere_fine.vertices, dtype=torch.float32))
        self.register_buffer('faces_fine',
                           torch.tensor(sphere_fine.faces, dtype=torch.long))
        
        # Deformation networks for each level
        # Coarse: Large deformations allowed
        self.coarse_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 162 * 3)
        )
        
        # Mid: Medium deformations
        self.mid_net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 642 * 3)
        )
        
        # Fine: Small deformations only
        self.fine_net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2562 * 3)
        )
        
        # Feature extractors for intermediate levels
        self.coarse_feat = nn.Linear(162 * 3, hidden_dim)
        self.mid_feat = nn.Linear(642 * 3, hidden_dim)
        
        # Initialize with small weights
        for net in [self.coarse_net, self.mid_net, self.fine_net]:
            nn.init.normal_(net[-1].weight, 0, 0.001)
            nn.init.zeros_(net[-1].bias)
    
    def forward(self, features):
        """
        Progressive mesh deformation: coarse -> mid -> fine
        Returns vertices at each level for multi-scale supervision
        """
        B = features.shape[0]
        
        # ===== COARSE LEVEL (162 vertices) =====
        offsets_coarse = self.coarse_net(features)
        offsets_coarse = offsets_coarse.view(B, 162, 3)
        offsets_coarse = torch.tanh(offsets_coarse) * 0.8  # Larger deformations allowed
        
        vertices_coarse_batch = self.vertices_coarse.unsqueeze(0).expand(B, -1, -1)
        deformed_coarse = vertices_coarse_batch + offsets_coarse
        
        # Extract features from coarse shape
        coarse_flat = deformed_coarse.view(B, -1)
        coarse_features = self.coarse_feat(coarse_flat)
        
        # ===== MID LEVEL (642 vertices) =====
        # Use upsampled coarse mesh as initialization
        mid_init = self._upsample_vertices(deformed_coarse, self.vertices_mid)
        
        mid_input = torch.cat([features, coarse_features], dim=1)
        offsets_mid = self.mid_net(mid_input)
        offsets_mid = offsets_mid.view(B, 642, 3)
        offsets_mid = torch.tanh(offsets_mid) * 0.3  # Medium deformations
        
        deformed_mid = mid_init + offsets_mid
        
        # Extract features from mid shape
        mid_flat = deformed_mid.view(B, -1)
        mid_features = self.mid_feat(mid_flat)
        
        # ===== FINE LEVEL (2562 vertices) =====
        # Use upsampled mid mesh as initialization
        fine_init = self._upsample_vertices(deformed_mid, self.vertices_fine)
        
        fine_input = torch.cat([features, mid_features], dim=1)
        offsets_fine = self.fine_net(fine_input)
        offsets_fine = offsets_fine.view(B, 2562, 3)
        offsets_fine = torch.tanh(offsets_fine) * 0.1  # Small refinements only
        
        deformed_fine = fine_init + offsets_fine
        
        return {
            'coarse': deformed_coarse,
            'mid': deformed_mid,
            'fine': deformed_fine,
            'faces_coarse': self.faces_coarse,
            'faces_mid': self.faces_mid,
            'faces_fine': self.faces_fine
        }
    
    def _upsample_vertices(self, coarse_verts, target_template):
        """
        Upsample coarse vertices to match target template size
        Uses nearest neighbor interpolation on the sphere
        """
        B = coarse_verts.shape[0]
        target_size = target_template.shape[0]
        
        # Normalize to unit sphere
        coarse_norm = coarse_verts / (torch.norm(coarse_verts, dim=2, keepdim=True) + 1e-8)
        target_norm = target_template.unsqueeze(0).expand(B, -1, -1)
        target_norm = target_norm / (torch.norm(target_norm, dim=2, keepdim=True) + 1e-8)
        
        # Find nearest neighbors on sphere (using dot product)
        similarity = torch.bmm(target_norm, coarse_norm.transpose(1, 2))  # (B, target, coarse)
        nearest_idx = torch.argmax(similarity, dim=2)  # (B, target)
        
        # Get radial distances from coarse mesh
        coarse_radii = torch.norm(coarse_verts, dim=2)  # (B, coarse)
        
        # Interpolate radii
        nearest_radii = torch.gather(coarse_radii, 1, nearest_idx)  # (B, target)
        
        # Scale target template by interpolated radii
        upsampled = target_norm * nearest_radii.unsqueeze(2)
        
        return upsampled


class ImprovedNormalDecoder(nn.Module):
    """Predict normals for the fine mesh"""
    def __init__(self, feature_dim=768, hidden_dim=1024):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2562 * 3)
        )

    def forward(self, features):
        B = features.shape[0]
        normals = self.decoder(features)
        normals = normals.view(B, 2562, 3)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        return normals


class ImprovedGeometryModel(nn.Module):
    """
    FIXED: Multi-scale geometry model with proper mesh handling
    """

    def __init__(self, freeze_encoder=False, hidden_dim=1024):
        super().__init__()
        
        print("="*70)
        print("INITIALIZING IMPROVED MULTI-SCALE GEOMETRY MODEL")
        print("="*70)

        self.image_encoder = MultiViewImageEncoder(
            model_name='facebook/dinov2-base',
            freeze=freeze_encoder
        )
        feature_dim = self.image_encoder.feature_dim

        self.view_aggregator = ViewAggregator(feature_dim=feature_dim)
        
        # NEW: Coarse-to-fine decoder
        self.mesh_decoder = CoarseToFineDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        self.normal_decoder = ImprovedNormalDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

        print(f"✓ Multi-scale model initialized")
        print(f"  Coarse: 162 vertices")
        print(f"  Mid: 642 vertices")
        print(f"  Fine: 2562 vertices")
        print(f"  Progressive refinement: YES")
        print("="*70)

    def forward(self, images_dict, return_all_levels=False):
        """
        Forward pass
        Args:
            images_dict: Dict of 6 view images
            return_all_levels: If True, return all resolution levels
        Returns:
            If return_all_levels=False: (fine_vertices, normals)
            If return_all_levels=True: mesh_outputs dict with all levels
        """
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        images_list = [images_dict[name] for name in view_names]
        
        # Encode views
        view_features = []
        for img in images_list:
            feat = self.image_encoder(img)
            view_features.append(feat)
        
        view_features = torch.stack(view_features, dim=1)
        aggregated = self.view_aggregator(view_features)
        
        # Decode mesh at multiple scales
        mesh_outputs = self.mesh_decoder(aggregated)
        
        # Predict normals for fine mesh
        normals = self.normal_decoder(aggregated)
        
        if return_all_levels:
            mesh_outputs['normals'] = normals
            return mesh_outputs
        else:
            return mesh_outputs['fine'], normals

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# IMPROVED LOSS FUNCTIONS WITH MULTI-SCALE SUPERVISION
# ============================================================================

def chamfer_distance(pred, gt):
    """Symmetric Chamfer distance"""
    pred_to_gt = torch.cdist(pred, gt, p=2)
    loss1 = torch.mean(torch.min(pred_to_gt, dim=1)[0])
    
    gt_to_pred = torch.cdist(gt, pred, p=2)
    loss2 = torch.mean(torch.min(gt_to_pred, dim=1)[0])
    
    return (loss1 + loss2) / 2


def mesh_edge_loss(vertices, faces):
    """
    CRITICAL: Compute edge length regularization using ACTUAL mesh edges
    This is much better than k-NN approximation
    """
    # Get edges from faces
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)
    
    # Compute edge lengths
    v1 = vertices[edges[:, 0]]
    v2 = vertices[edges[:, 1]]
    edge_lengths = torch.norm(v2 - v1, dim=1)
    
    # Target: uniform edge length
    mean_length = edge_lengths.mean()
    variance = torch.mean((edge_lengths - mean_length) ** 2)
    
    # Penalize variance + prevent collapse
    collapse_penalty = torch.relu(0.01 - edge_lengths).mean()
    
    return variance + 10.0 * collapse_penalty


def mesh_laplacian_loss(vertices, faces):
    """
    CRITICAL: Laplacian smoothness using ACTUAL mesh connectivity
    """
    # Build adjacency
    num_verts = vertices.shape[0]
    adjacency = {}
    for i in range(num_verts):
        adjacency[i] = set()
    
    for face in faces:
        for i in range(3):
            v1, v2 = face[i].item(), face[(i+1)%3].item()
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
    
    # Compute Laplacian
    laplacian_loss = 0.0
    for i in range(num_verts):
        if len(adjacency[i]) > 0:
            neighbors = torch.stack([vertices[j] for j in adjacency[i]])
            neighbor_mean = neighbors.mean(dim=0)
            laplacian_loss += torch.sum((vertices[i] - neighbor_mean) ** 2)
    
    return laplacian_loss / num_verts


def mesh_normal_consistency(vertices, faces):
    """Compute normal consistency using mesh faces"""
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = torch.cross(e1, e2, dim=1)
    face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
    
    # Compute vertex normals by averaging adjacent face normals
    num_verts = vertices.shape[0]
    vertex_normals = torch.zeros_like(vertices)
    counts = torch.zeros(num_verts, device=vertices.device)
    
    for i in range(faces.shape[0]):
        for j in range(3):
            v_idx = faces[i, j]
            vertex_normals[v_idx] += face_normals[i]
            counts[v_idx] += 1
    
    vertex_normals = vertex_normals / (counts.unsqueeze(1) + 1e-8)
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
    
    # Consistency: adjacent vertices should have similar normals
    consistency_loss = 0.0
    for face in faces:
        n0, n1, n2 = vertex_normals[face[0]], vertex_normals[face[1]], vertex_normals[face[2]]
        consistency_loss += (1 - torch.dot(n0, n1)) + (1 - torch.dot(n1, n2)) + (1 - torch.dot(n2, n0))
    
    return consistency_loss / (faces.shape[0] * 3)


def multi_scale_geometry_loss(
    mesh_outputs,
    gt_points,
    lambda_chamfer=1.0,
    lambda_edge=2.0,      # MUCH HIGHER
    lambda_smooth=1.0,    # MUCH HIGHER
    lambda_normal=0.5
):
    """
    Multi-scale loss with proper mesh regularization
    """
    total_loss = 0.0
    loss_dict = {}
    
    # Coarse level (weight: 0.3)
    coarse_verts = mesh_outputs['coarse'][0]
    coarse_faces = mesh_outputs['faces_coarse']
    
    loss_chamfer_coarse = chamfer_distance(coarse_verts, gt_points)
    loss_edge_coarse = mesh_edge_loss(coarse_verts, coarse_faces)
    loss_smooth_coarse = mesh_laplacian_loss(coarse_verts, coarse_faces)
    loss_normal_coarse = mesh_normal_consistency(coarse_verts, coarse_faces)
    
    coarse_loss = (
        lambda_chamfer * loss_chamfer_coarse +
        lambda_edge * loss_edge_coarse +
        lambda_smooth * loss_smooth_coarse +
        lambda_normal * loss_normal_coarse
    )
    total_loss += 0.3 * coarse_loss
    
    loss_dict['chamfer_coarse'] = loss_chamfer_coarse.item()
    loss_dict['edge_coarse'] = loss_edge_coarse.item()
    
    # Mid level (weight: 0.3)
    mid_verts = mesh_outputs['mid'][0]
    mid_faces = mesh_outputs['faces_mid']
    
    loss_chamfer_mid = chamfer_distance(mid_verts, gt_points)
    loss_edge_mid = mesh_edge_loss(mid_verts, mid_faces)
    loss_smooth_mid = mesh_laplacian_loss(mid_verts, mid_faces)
    loss_normal_mid = mesh_normal_consistency(mid_verts, mid_faces)
    
    mid_loss = (
        lambda_chamfer * loss_chamfer_mid +
        lambda_edge * loss_edge_mid +
        lambda_smooth * loss_smooth_mid +
        lambda_normal * loss_normal_mid
    )
    total_loss += 0.3 * mid_loss
    
    loss_dict['chamfer_mid'] = loss_chamfer_mid.item()
    loss_dict['edge_mid'] = loss_edge_mid.item()
    
    # Fine level (weight: 0.4 - most important)
    fine_verts = mesh_outputs['fine'][0]
    fine_faces = mesh_outputs['faces_fine']
    
    loss_chamfer_fine = chamfer_distance(fine_verts, gt_points)
    loss_edge_fine = mesh_edge_loss(fine_verts, fine_faces)
    loss_smooth_fine = mesh_laplacian_loss(fine_verts, fine_faces)
    loss_normal_fine = mesh_normal_consistency(fine_verts, fine_faces)
    
    fine_loss = (
        lambda_chamfer * loss_chamfer_fine +
        lambda_edge * loss_edge_fine +
        lambda_smooth * loss_smooth_fine +
        lambda_normal * loss_normal_fine
    )
    total_loss += 0.4 * fine_loss
    
    loss_dict['chamfer_fine'] = loss_chamfer_fine.item()
    loss_dict['edge_fine'] = loss_edge_fine.item()
    loss_dict['smooth_fine'] = loss_smooth_fine.item()
    loss_dict['normal_fine'] = loss_normal_fine.item()
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict