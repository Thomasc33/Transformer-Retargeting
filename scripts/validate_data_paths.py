#!/usr/bin/env python3
"""
Data Path Validation Script for Transformer Retargeting Project

This script helps users understand and validate the correct data directory structure
for NTU RGB+D datasets, including the specific requirements for NTU120 which uses
both NTU60 and NTU120 skeleton data.

Usage:
    python scripts/validate_data_paths.py
    python scripts/validate_data_paths.py --fix-structure
    python scripts/validate_data_paths.py --show-expected
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_config(config_path: str = "configs/main_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

def show_expected_structure():
    """Show the expected data directory structure."""
    print("\n📁 EXPECTED DATA DIRECTORY STRUCTURE")
    print("=" * 50)
    
    print("""
📁 data/
├── 📁 nturgbd_raw/                    # Main raw data directory
│   ├── 📁 nturgb+d_skeletons/         # NTU RGB+D 60 skeleton files
│   │   ├── S001C001P001R001A001.skeleton
│   │   ├── S001C001P001R001A002.skeleton
│   │   └── ... (56,880 files for NTU60)
│   │
│   ├── 📁 nturgb+d_skeletons120/      # NTU RGB+D 120 skeleton files
│   │   ├── S001C001P001R001A061.skeleton
│   │   ├── S001C001P001R001A062.skeleton
│   │   └── ... (57,600 additional files for NTU120)
│   │
│   └── 📁 [other directories like videos, etc.]
│
├── 📁 etri_raw/                       # ETRI raw data (if using ETRI)
│   ├── A001P001G001C001.json
│   └── ... (ETRI JSON files)
│
├── 📁 ntu/                           # Processed NTU60 data (created by preprocessing)
├── 📁 ntu120/                        # Processed NTU120 data (created by preprocessing)
└── 📁 etri/                          # Processed ETRI data (created by preprocessing)
""")
    
    print("🔍 KEY POINTS:")
    print("• NTU60 uses only 'nturgb+d_skeletons' directory")
    print("• NTU120 uses BOTH 'nturgb+d_skeletons' AND 'nturgb+d_skeletons120' directories")
    print("• All raw data should be in 'data/nturgbd_raw/'")
    print("• Processed data will be created in 'data/ntu/', 'data/ntu120/', etc.")

def validate_data_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the current data directory structure."""
    print("\n🔍 VALIDATING DATA STRUCTURE")
    print("=" * 40)
    
    results = {
        'ntu': {'status': 'unknown', 'files': 0, 'issues': []},
        'ntu120': {'status': 'unknown', 'files': 0, 'issues': []},
        'etri': {'status': 'unknown', 'files': 0, 'issues': []}
    }
    
    # Check NTU60
    print("\n📊 Checking NTU RGB+D 60...")
    ntu_config = config['datasets']['ntu']
    raw_path = Path(ntu_config['raw_data_path'])
    skeleton_dir = raw_path / ntu_config['skeleton_dir']
    
    if not raw_path.exists():
        results['ntu']['status'] = 'missing_raw_dir'
        results['ntu']['issues'].append(f"Raw data directory not found: {raw_path}")
        print(f"  ❌ Raw data directory not found: {raw_path}")
    elif not skeleton_dir.exists():
        results['ntu']['status'] = 'missing_skeleton_dir'
        results['ntu']['issues'].append(f"Skeleton directory not found: {skeleton_dir}")
        print(f"  ❌ Skeleton directory not found: {skeleton_dir}")
    else:
        skeleton_files = list(skeleton_dir.glob('*.skeleton'))
        results['ntu']['files'] = len(skeleton_files)
        
        if len(skeleton_files) == 0:
            results['ntu']['status'] = 'no_files'
            results['ntu']['issues'].append("No .skeleton files found")
            print(f"  ❌ No .skeleton files found in {skeleton_dir}")
        elif len(skeleton_files) < 50000:  # Expected ~56,880 for NTU60
            results['ntu']['status'] = 'incomplete'
            results['ntu']['issues'].append(f"Only {len(skeleton_files)} files found, expected ~56,880")
            print(f"  ⚠️  Only {len(skeleton_files)} files found, expected ~56,880")
        else:
            results['ntu']['status'] = 'good'
            print(f"  ✅ Found {len(skeleton_files)} skeleton files")
    
    # Check NTU120
    print("\n📊 Checking NTU RGB+D 120...")
    ntu120_config = config['datasets']['ntu120']
    ntu120_skeleton_dir = raw_path / ntu120_config['skeleton_dir']
    ntu60_skeleton_dir = raw_path / ntu120_config['ntu60_skeleton_dir']
    
    ntu120_files = []
    ntu60_files = []
    
    if ntu120_skeleton_dir.exists():
        ntu120_files = list(ntu120_skeleton_dir.glob('*.skeleton'))
        print(f"  📁 Found {len(ntu120_files)} files in {ntu120_skeleton_dir}")
    else:
        results['ntu120']['issues'].append(f"NTU120 skeleton directory not found: {ntu120_skeleton_dir}")
        print(f"  ❌ NTU120 skeleton directory not found: {ntu120_skeleton_dir}")
    
    if ntu60_skeleton_dir.exists():
        ntu60_files = list(ntu60_skeleton_dir.glob('*.skeleton'))
        print(f"  📁 Found {len(ntu60_files)} files in {ntu60_skeleton_dir}")
    else:
        results['ntu120']['issues'].append(f"NTU60 skeleton directory not found: {ntu60_skeleton_dir}")
        print(f"  ❌ NTU60 skeleton directory not found: {ntu60_skeleton_dir}")
    
    total_ntu120_files = len(ntu120_files) + len(ntu60_files)
    results['ntu120']['files'] = total_ntu120_files
    
    if total_ntu120_files == 0:
        results['ntu120']['status'] = 'no_files'
        results['ntu120']['issues'].append("No .skeleton files found for NTU120")
        print(f"  ❌ No .skeleton files found for NTU120")
    elif total_ntu120_files < 100000:  # Expected ~114,480 total for NTU120
        results['ntu120']['status'] = 'incomplete'
        results['ntu120']['issues'].append(f"Only {total_ntu120_files} total files found, expected ~114,480")
        print(f"  ⚠️  Only {total_ntu120_files} total files found, expected ~114,480")
    else:
        results['ntu120']['status'] = 'good'
        print(f"  ✅ Found {total_ntu120_files} total skeleton files for NTU120")
    
    # Check ETRI
    print("\n📊 Checking ETRI...")
    etri_config = config['datasets']['etri']
    etri_raw_path = Path(etri_config['raw_data_path'])
    
    if not etri_raw_path.exists():
        results['etri']['status'] = 'missing_raw_dir'
        results['etri']['issues'].append(f"ETRI raw data directory not found: {etri_raw_path}")
        print(f"  ❌ ETRI raw data directory not found: {etri_raw_path}")
    else:
        json_files = list(etri_raw_path.glob('*.json'))
        results['etri']['files'] = len(json_files)
        
        if len(json_files) == 0:
            results['etri']['status'] = 'no_files'
            results['etri']['issues'].append("No .json files found")
            print(f"  ❌ No .json files found in {etri_raw_path}")
        else:
            results['etri']['status'] = 'good'
            print(f"  ✅ Found {len(json_files)} JSON files")
    
    return results

def print_validation_summary(results: Dict[str, Any]):
    """Print a summary of validation results."""
    print("\n📋 VALIDATION SUMMARY")
    print("=" * 30)
    
    for dataset, result in results.items():
        status = result['status']
        files = result['files']
        issues = result['issues']
        
        if status == 'good':
            print(f"✅ {dataset.upper()}: Ready ({files:,} files)")
        elif status == 'incomplete':
            print(f"⚠️  {dataset.upper()}: Incomplete ({files:,} files)")
        elif status in ['missing_raw_dir', 'missing_skeleton_dir', 'no_files']:
            print(f"❌ {dataset.upper()}: Not available")
        else:
            print(f"❓ {dataset.upper()}: Unknown status")
        
        for issue in issues:
            print(f"   • {issue}")

def provide_setup_guidance(results: Dict[str, Any]):
    """Provide guidance on how to set up the data correctly."""
    print("\n💡 SETUP GUIDANCE")
    print("=" * 20)
    
    has_issues = any(result['status'] != 'good' for result in results.values())
    
    if not has_issues:
        print("🎉 All datasets are properly configured!")
        print("\n📋 Next steps:")
        print("1. Run preprocessing: python scripts/preprocess.py --interactive")
        print("2. Run sampling: python scripts/sample.py --interactive")
        print("3. Start training: python scripts/pipeline.py --interactive")
        return
    
    print("🔧 To fix data structure issues:")
    print()
    
    if results['ntu']['status'] != 'good' or results['ntu120']['status'] != 'good':
        print("📊 For NTU RGB+D datasets:")
        print("1. Download NTU RGB+D and/or NTU RGB+D 120 datasets")
        print("2. Extract skeleton files to:")
        print("   • data/nturgbd_raw/nturgb+d_skeletons/ (for NTU60)")
        print("   • data/nturgbd_raw/nturgb+d_skeletons120/ (for NTU120)")
        print("3. Ensure both directories exist for NTU120 (it uses both)")
        print()
    
    if results['etri']['status'] != 'good':
        print("📊 For ETRI dataset:")
        print("1. Download ETRI dataset")
        print("2. Extract JSON files to: data/etri_raw/")
        print()
    
    print("🔍 Verify structure with: python scripts/validate_data_paths.py")
    print("📚 See docs/README.md for detailed setup instructions")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Data Path Validation for Transformer Retargeting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--show-expected', action='store_true',
                       help='Show the expected data directory structure')
    parser.add_argument('--config', type=str, default='configs/main_config.yaml',
                       help='Path to main configuration file')
    
    args = parser.parse_args()
    
    print("🔍 TRANSFORMER RETARGETING DATA VALIDATION")
    print("=" * 50)
    
    if args.show_expected:
        show_expected_structure()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Show expected structure
    show_expected_structure()
    
    # Validate current structure
    results = validate_data_structure(config)
    
    # Print summary
    print_validation_summary(results)
    
    # Provide guidance
    provide_setup_guidance(results)

if __name__ == "__main__":
    main()
