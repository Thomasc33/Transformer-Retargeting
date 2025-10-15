#!/usr/bin/env python3
"""
Validation script to test the autoregressive training optimizations.
Run this before deploying to full-scale training.
"""

import os
import sys
import time
import torch
import subprocess
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def run_test(test_name, cmd, expected_time_minutes=None):
    """Run a test command and measure performance."""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"Expected time: {expected_time_minutes} minutes" if expected_time_minutes else "Expected time: Unknown")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        print(f"\n✅ Test completed in {duration_minutes:.2f} minutes")
        
        if expected_time_minutes and duration_minutes > expected_time_minutes * 1.5:
            print(f"⚠️  WARNING: Test took longer than expected ({duration_minutes:.2f} > {expected_time_minutes * 1.5:.2f} minutes)")
        elif expected_time_minutes and duration_minutes < expected_time_minutes * 0.5:
            print(f"🚀 EXCELLENT: Test completed much faster than expected!")
        
        # Check for errors
        if result.returncode != 0:
            print(f"❌ Test failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, duration_minutes
        
        # Look for key performance indicators in output
        output = result.stdout
        if "📊 Batch" in output:
            print("✅ Found batch progress indicators")
        if "GPU Memory:" in output:
            print("✅ Found GPU memory monitoring")
        if "PERFORMANCE PROFILING SUMMARY" in output:
            print("✅ Found profiling output")
        
        return True, duration_minutes
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 1 hour")
        return False, 60.0
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False, 0.0

def check_gpu_availability():
    """Check if GPUs are available and ready."""
    print("🔍 Checking GPU availability...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate autoregressive training optimizations")
    parser.add_argument('--quick', action='store_true', help='Run only quick tests')
    parser.add_argument('--full', action='store_true', help='Run full validation suite')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    
    print("🚀 AUTOREGRESSIVE TRAINING OPTIMIZATION VALIDATOR")
    print("="*60)
    
    # Check prerequisites
    if not check_gpu_availability():
        print("❌ GPU check failed. Cannot proceed.")
        return 1
    
    # Test configurations
    tests = []
    
    if args.quick or not args.full:
        print("\n📋 QUICK VALIDATION TESTS")
        tests.extend([
            {
                'name': 'Micro Test (100 samples, 1 epoch)',
                'cmd': f'python main.py --dataset ntu --setting cv --epochs 1 --train-samples 100 --test-samples 50 --gpus {args.gpus} --batch-size 4',
                'expected_minutes': 2
            },
            {
                'name': 'Small Test (1000 samples, 1 epoch)',
                'cmd': f'python main.py --dataset ntu --setting cv --epochs 1 --train-samples 1000 --test-samples 100 --gpus {args.gpus} --batch-size 8',
                'expected_minutes': 5
            }
        ])
    
    if args.full:
        print("\n📋 COMPREHENSIVE VALIDATION TESTS")
        tests.extend([
            {
                'name': 'Medium Test (10k samples, 2 epochs)',
                'cmd': f'python main.py --dataset ntu --setting cv --epochs 2 --train-samples 10000 --test-samples 1000 --gpus {args.gpus} --batch-size 16',
                'expected_minutes': 20
            },
            {
                'name': 'Large Test (50k samples, 2 epochs)',
                'cmd': f'python main.py --dataset ntu --setting cv --epochs 2 --train-samples 50000 --test-samples 5000 --gpus {args.gpus} --batch-size 16',
                'expected_minutes': 60
            }
        ])
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for test in tests:
        success, duration = run_test(test['name'], test['cmd'], test.get('expected_minutes'))
        results.append({
            'name': test['name'],
            'success': success,
            'duration': duration,
            'expected': test.get('expected_minutes', 0)
        })
        
        if not success:
            print(f"\n❌ Test '{test['name']}' failed. Consider investigating before proceeding.")
            break
    
    # Summary
    total_duration = (time.time() - total_start_time) / 60
    print(f"\n{'='*60}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total validation time: {total_duration:.2f} minutes")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Optimizations are working correctly")
        print("🚀 Ready for full-scale training")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        for result in results:
            if result['success'] and result['expected'] > 0:
                speedup = result['expected'] / result['duration']
                if speedup > 1.2:
                    print(f"   {result['name']}: {speedup:.1f}x faster than expected! 🚀")
                elif speedup > 0.8:
                    print(f"   {result['name']}: Within expected range ({speedup:.1f}x)")
                else:
                    print(f"   {result['name']}: Slower than expected ({speedup:.1f}x) ⚠️")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Deploy optimizations to full training")
        print("2. Monitor first 24 hours closely")
        print("3. Check profiling output for any remaining bottlenecks")
        print("4. Adjust parameters based on actual performance")
        
        return 0
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")
        print("🔧 Please investigate issues before deploying to full-scale training")
        
        print(f"\n🔍 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Check GPU memory usage during failed tests")
        print("2. Verify all dependencies are installed")
        print("3. Check for CUDA/PyTorch compatibility issues")
        print("4. Review error messages in test output")
        
        return 1

if __name__ == '__main__':
    exit(main())
