#!/bin/bash
# Install Hunyuan3D for Knowledge Distillation

echo "======================================================================"
echo "INSTALLING HUNYUAN3D FOR KNOWLEDGE DISTILLATION"
echo "======================================================================"

# Step 1: Install dependencies
echo "Installing dependencies..."
pip install huggingface_hub

# Step 2: Download Hunyuan3D-2mv (multiview model)
echo "Downloading Hunyuan3D-2mv model..."
mkdir -p weights/hunyuan3d
huggingface-cli download tencent/Hunyuan3D-2mv \
    --local-dir ./weights/hunyuan3d/2mv \
    --include "hunyuan3d-dit-v2-mv/*"

# Step 3: Download turbo version (faster)
echo "Downloading Hunyuan3D-2mv-turbo (faster version)..."
huggingface-cli download tencent/Hunyuan3D-2mv \
    --local-dir ./weights/hunyuan3d/2mv-turbo \
    --include "hunyuan3d-dit-v2-mv-turbo/*"

# Step 4: Install hy3dgen library
echo "Installing hy3dgen..."
pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git

# Step 5: Install additional dependencies
echo "Installing additional dependencies..."
pip install trimesh einops diffusers accelerate

echo ""
echo "======================================================================"
echo "✓ Hunyuan3D installed successfully!"
echo "======================================================================"
echo ""
echo "Models downloaded to:"
echo "  - Standard: ./weights/hunyuan3d/2mv"
echo "  - Turbo:    ./weights/hunyuan3d/2mv-turbo"
echo ""
echo "Next steps:"
echo "  1. Test teacher model: python test_hunyuan_teacher.py"
echo "  2. Generate teacher predictions: python generate_teacher_cache.py"
echo "  3. Train with distillation: python train_with_distillation.py"
