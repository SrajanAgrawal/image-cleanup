import torch
import mlx.core as mx

print("--- Hardware Check ---")
# Check PyTorch Metal support
if torch.backends.mps.is_available():
    print("✅ PyTorch can see your GPU (MPS).")
else:
    print("❌ PyTorch is stuck on CPU. Check Miniforge install.")

# Check MLX support
print(f"✅ MLX is using: {mx.default_device()}")