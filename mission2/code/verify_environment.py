#!/usr/bin/env python3
"""
Environment Verification Script for Zen Garden Robot

This script verifies that all required dependencies are installed and
properly configured for the MI300X cluster and SmolVLA training.

Usage:
    python verify_environment.py
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  WARNING: Python 3.10+ recommended")
        return False
    print("  OK")
    return True


def check_pytorch():
    """Check PyTorch and ROCm availability."""
    print("\nChecking PyTorch and ROCm...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA/ROCm available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"  Device count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  Device {i}: {device_name}")
                
                # Check memory
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    Memory: {total_mem:.1f} GB")
            
            # Test tensor operation
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.mm(x, x)
            print(f"  GPU tensor test: OK")
        else:
            print("  WARNING: No GPU available. Training will be slow.")
            return False
        
        print("  OK")
        return True
        
    except ImportError:
        print("  ERROR: PyTorch not installed")
        print("  Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_lerobot():
    """Check LeRobot installation."""
    print("\nChecking LeRobot...")
    try:
        import lerobot
        print(f"  LeRobot version: {getattr(lerobot, '__version__', 'unknown')}")
        
        # Check key modules
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        print("  LeRobotDataset: OK")
        
        from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
        print("  OpenCVCamera: OK")
        
        print("  OK")
        return True
        
    except ImportError as e:
        print(f"  ERROR: LeRobot not installed or missing dependencies")
        print(f"  Details: {e}")
        print("  Install with:")
        print("    git clone https://github.com/huggingface/lerobot")
        print("    pip install -e ./lerobot")
        return False


def check_smolvla():
    """Check SmolVLA installation."""
    print("\nChecking SmolVLA...")
    try:
        # Try importing SmolVLA components
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print("  Transformers: OK")
        
        # Check if SmolVLA model can be loaded
        try:
            processor = AutoProcessor.from_pretrained(
                "HuggingFaceTB/SmolVLM-Instruct",
                trust_remote_code=True
            )
            print("  SmolVLM processor: OK")
        except Exception as e:
            print(f"  SmolVLM processor: WARNING - {e}")
        
        print("  OK (basic check passed)")
        return True
        
    except ImportError as e:
        print(f"  ERROR: {e}")
        print("  Install with:")
        print("    git clone https://github.com/huggingface/smolvla")
        print("    pip install -e ./smolvla")
        return False


def check_opencv():
    """Check OpenCV installation."""
    print("\nChecking OpenCV...")
    try:
        import cv2
        print(f"  OpenCV version: {cv2.__version__}")
        
        # Check if camera access works
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  Camera access: OK")
            cap.release()
        else:
            print("  Camera access: WARNING - No camera found (may be OK on headless server)")
        
        print("  OK")
        return True
        
    except ImportError:
        print("  ERROR: OpenCV not installed")
        print("  Install with: pip install opencv-python")
        return False


def check_additional_deps():
    """Check additional dependencies."""
    print("\nChecking additional dependencies...")
    
    deps = {
        "numpy": "numpy",
        "PIL": "pillow",
        "wandb": "wandb",
        "accelerate": "accelerate",
        "anthropic": "anthropic",
    }
    
    all_ok = True
    for module_name, package_name in deps.items():
        try:
            __import__(module_name.split(".")[0])
            print(f"  {module_name}: OK")
        except ImportError:
            print(f"  {module_name}: NOT INSTALLED (pip install {package_name})")
            all_ok = False
    
    return all_ok


def check_directory_structure():
    """Check expected directory structure."""
    print("\nChecking directory structure...")
    
    base_dir = Path(__file__).parent.parent
    expected_dirs = [
        "code",
        "wandb",
    ]
    
    recommended_dirs = [
        "data",
        "goals",
        "checkpoints",
        "configs",
    ]
    
    all_ok = True
    
    for dir_name in expected_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  {dir_name}/: OK")
        else:
            print(f"  {dir_name}/: MISSING")
            all_ok = False
    
    print("\n  Recommended directories:")
    for dir_name in recommended_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"    {dir_name}/: OK")
        else:
            print(f"    {dir_name}/: NOT CREATED (will be created automatically)")
            # Create the directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"      -> Created {dir_name}/")
    
    return all_ok


def check_lerobot_record():
    """Check if lerobot-record command is available."""
    print("\nChecking LeRobot recording tools...")
    try:
        result = subprocess.run(
            ["lerobot-record", "--help"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  lerobot-record: OK")
            return True
        else:
            print("  lerobot-record: ERROR")
            return False
    except FileNotFoundError:
        print("  lerobot-record: NOT FOUND")
        print("  Alternative: python -m lerobot.scripts.control_robot record")
        return False


def main():
    print("=" * 60)
    print("ZEN GARDEN ROBOT - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & ROCm", check_pytorch),
        ("LeRobot", check_lerobot),
        ("SmolVLA", check_smolvla),
        ("OpenCV", check_opencv),
        ("Additional Dependencies", check_additional_deps),
        ("Directory Structure", check_directory_structure),
        ("LeRobot Recording Tools", check_lerobot_record),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    critical_checks = ["Python Version", "PyTorch & ROCm", "LeRobot"]
    
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        critical = " (CRITICAL)" if name in critical_checks and not passed else ""
        print(f"  {name}: {status}{critical}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks passed! Environment is ready.")
    else:
        print("Some checks failed. Please install missing dependencies.")
        failed_critical = [name for name in critical_checks if not results.get(name, False)]
        if failed_critical:
            print(f"\nCRITICAL: {', '.join(failed_critical)} must be fixed before proceeding.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
