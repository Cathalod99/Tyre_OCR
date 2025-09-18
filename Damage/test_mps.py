# test_mps.py - Test MPS availability and performance
import torch
import time
import numpy as np

def test_device_performance():
    """Test performance difference between CPU and MPS"""
    
    print("🔍 DEVICE AVAILABILITY TEST")
    print("=" * 40)
    
    # Check device availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) is available!")
    elif torch.cuda.is_available():
        print("✅ CUDA (NVIDIA GPU) is available!")
    else:
        print("⚠️ Only CPU available")
    
    # Test performance with a simple operation
    print(f"\n⚡ PERFORMANCE TEST")
    print("-" * 30)
    
    # Create test data
    size = (1000, 1000)
    x = torch.randn(size)
    
    # Test CPU
    start_time = time.time()
    for _ in range(10):
        y = torch.matmul(x, x)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.3f}s")
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x_mps = x.to(device)
        
        start_time = time.time()
        for _ in range(10):
            y = torch.matmul(x_mps, x_mps)
        mps_time = time.time() - start_time
        print(f"MPS time: {mps_time:.3f}s")
        
        speedup = cpu_time / mps_time
        print(f"🚀 MPS speedup: {speedup:.1f}x faster!")
        
        if speedup > 2:
            print("✅ Significant speedup expected for training!")
        else:
            print("⚠️ Modest speedup - may vary with model size")
    
    # Show recommended device
    print(f"\n🎯 RECOMMENDED DEVICE:")
    if torch.backends.mps.is_available():
        print("✅ Use MPS (Apple Silicon) - 3-6x faster than CPU")
    elif torch.cuda.is_available():
        print("✅ Use CUDA (NVIDIA GPU) - 5-10x faster than CPU")
    else:
        print("⚠️ Use CPU - no GPU acceleration available")

if __name__ == "__main__":
    test_device_performance()
