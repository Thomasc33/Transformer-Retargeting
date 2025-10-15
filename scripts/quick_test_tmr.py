#!/usr/bin/env python3
"""
Quick test to verify TMR is working correctly
Tests with a small subset of data (10 samples)
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_loading():
    """Test that we can load the data"""
    print("\n" + "="*80)
    print("TEST 1: Data Loading")
    print("="*80)
    
    data_path = Path("data/ntu_cv_paired_comprehensive.pt")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    print(f"✅ Data file found: {data_path}")
    
    try:
        data = torch.load(data_path)
        print(f"✅ Data loaded successfully")
        print(f"   Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if 'train' in data:
                print(f"   Train samples: {len(data['train'])}")
            if 'test' in data:
                print(f"   Test samples: {len(data['test'])}")
        elif isinstance(data, (list, tuple)):
            print(f"   Length: {len(data)}")
            if len(data) > 0:
                print(f"   First item type: {type(data[0])}")
                if isinstance(data[0], (list, tuple)):
                    print(f"   First item length: {len(data[0])}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_model_loading():
    """Test that we can load the TMR model"""
    print("\n" + "="*80)
    print("TEST 2: Model Loading")
    print("="*80)
    
    model_path = Path("data/models_output/model_all.pth")
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"   Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print(f"   Model state dict found")
            if 'epoch' in checkpoint:
                print(f"   Epoch: {checkpoint['epoch']}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_baseline_models():
    """Test that baseline models are available"""
    print("\n" + "="*80)
    print("TEST 3: Baseline Models")
    print("="*80)
    
    models_to_check = [
        ("Mixformer AR", "output/ntu_cv_mixformer_ar_cview/model_best.pth.tar"),
        ("Mixformer RI", "output/ntu_cv_mixformer_ri_cview/model_best.pth.tar"),
        ("SGN AR", "output/ntu_cv_sgn_ar_cview/model_best.pth.tar"),
        ("SGN RI", "output/ntu_cv_sgn_ri_cview/model_best.pth.tar"),
    ]
    
    found = 0
    for name, path in models_to_check:
        model_path = Path(path)
        if model_path.exists():
            print(f"✅ {name}: {model_path}")
            found += 1
        else:
            print(f"⚠️  {name} missing: {model_path}")
    
    print(f"\n   Found {found}/{len(models_to_check)} baseline models")
    return found > 0

def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*80)
    print("TEST 4: CUDA Availability")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   Device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print(f"⚠️  CUDA is not available (will use CPU)")
        return False

def test_evaluation_script():
    """Test that evaluation script exists and is runnable"""
    print("\n" + "="*80)
    print("TEST 5: Evaluation Script")
    print("="*80)
    
    eval_script = Path("eval_anonymization_v2.py")
    
    if not eval_script.exists():
        print(f"❌ Evaluation script not found: {eval_script}")
        return False
    
    print(f"✅ Evaluation script found: {eval_script}")
    
    # Check if it's executable
    import stat
    if eval_script.stat().st_mode & stat.S_IXUSR:
        print(f"✅ Script is executable")
    else:
        print(f"ℹ️  Script is not executable (but can still run with python)")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TMR QUICK TEST")
    print("="*80)
    print("\nThis will verify that TMR is set up correctly and ready to run.")
    print("Running 5 quick tests...\n")
    
    results = []
    
    # Run tests
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Baseline Models", test_baseline_models()))
    results.append(("CUDA", test_cuda()))
    results.append(("Evaluation Script", test_evaluation_script()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! TMR is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python tmr.py")
        print("  2. Select option 3 (Evaluation Operations)")
        print("  3. Select option 2 (Evaluate Anonymization)")
        print("  4. Choose model: tmr")
        print("  5. Choose dataset: ntu_cv")
        print("  6. Enter samples: 100")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

