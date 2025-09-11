"""
Parse SLURM output files for additional ablation study results.
This script extracts evaluation metrics from slurm output files.
"""

import os
import re
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class SlurmAblationParser:
    """Parse ablation study results from SLURM output files."""
    
    def __init__(self, base_dir="/users/tcarr23/Transformer-Retargeting"):
        self.base_dir = Path(base_dir)
        self.slurm_dir = self.base_dir / "slurm_out" / "experiments" / "ablations"
        self.output_dir = self.base_dir / "evaluation_suite" / "results" / "experiments" / "loss_ablation"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_slurm_file(self, file_path):
        """Parse a single SLURM output file for metrics."""
        print(f"📄 Parsing {file_path.name}...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract model configuration
        model_info = self.extract_model_info(content)
        
        # Extract evaluation metrics
        metrics = self.extract_metrics(content)
        
        if metrics:
            metrics.update(model_info)
            return metrics
        else:
            print(f"⚠️  No metrics found in {file_path.name}")
            return None
    
    def extract_model_info(self, content):
        """Extract model configuration information."""
        info = {
            'slurm_job': None,
            'model_type': 'Unknown',
            'ablated_component': 'Unknown',
            'dataset': 'ntu_cv'
        }
        
        # Extract SLURM job ID
        job_match = re.search(r'slurm-(\d+)\.out', content)
        if job_match:
            info['slurm_job'] = job_match.group(1)
        
        # Look for ablation indicators in the content
        ablation_patterns = {
            'bone_length': r'(?i)bone.?length|blc',
            'foot_contact': r'(?i)foot.?contact|fcc',
            'joint_limit': r'(?i)joint.?limit|jal',
            'velocity': r'(?i)velocity|fid',
            'end_effector': r'(?i)end.?effector',
            'smoothing': r'(?i)smooth|temporal',
            'full_model': r'(?i)full.?model|complete|baseline'
        }
        
        for component, pattern in ablation_patterns.items():
            if re.search(pattern, content):
                info['ablated_component'] = component
                break
        
        return info
    
    def extract_metrics(self, content):
        """Extract evaluation metrics from content."""
        metrics = {}
        
        # Common metric patterns
        patterns = {
            'accuracy': [
                r'Action Recognition Accuracy[:\s]+([0-9.]+)%?',
                r'AR[:\s]+([0-9.]+)%?',
                r'Accuracy[:\s]+([0-9.]+)%?'
            ],
            'identity_accuracy': [
                r'Re-identification Accuracy[:\s]+([0-9.]+)%?',
                r'RI[:\s]+([0-9.]+)%?',
                r'Identity Accuracy[:\s]+([0-9.]+)%?'
            ],
            'mse': [
                r'MSE[:\s]+([0-9.]+)',
                r'Mean Squared Error[:\s]+([0-9.]+)'
            ],
            'bone_length_error': [
                r'Bone Length Error[:\s]+([0-9.]+)',
                r'BLC[:\s]+([0-9.]+)'
            ],
            'foot_contact_error': [
                r'Foot Contact Error[:\s]+([0-9.]+)',
                r'FCC[:\s]+([0-9.]+)'
            ],
            'joint_angle_loss': [
                r'Joint Angle Loss[:\s]+([0-9.]+)',
                r'JAL[:\s]+([0-9.]+)'
            ],
            'temporal_smoothness': [
                r'Temporal Smoothness[:\s]+([0-9.]+)',
                r'TS[:\s]+([0-9.]+)'
            ],
            'velocity_consistency': [
                r'Velocity Consistency[:\s]+([0-9.]+)',
                r'VC[:\s]+([0-9.]+)'
            ]
        }
        
        # Extract metrics using patterns
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        metrics[metric] = value
                        break
                    except ValueError:
                        continue
        
        # Look for JSON-formatted results
        json_matches = re.findall(r'\{[^{}]*"accuracy"[^{}]*\}', content)
        for json_str in json_matches:
            try:
                json_data = json.loads(json_str)
                for key, value in json_data.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
            except json.JSONDecodeError:
                continue
        
        return metrics
    
    def parse_all_slurm_files(self):
        """Parse all SLURM files in the ablations directory."""
        if not self.slurm_dir.exists():
            print(f"❌ SLURM directory not found: {self.slurm_dir}")
            return None
        
        slurm_files = list(self.slurm_dir.glob("slurm-*.out"))
        if not slurm_files:
            print(f"❌ No SLURM files found in: {self.slurm_dir}")
            return None
        
        print(f"🔍 Found {len(slurm_files)} SLURM files to parse")
        
        all_results = []
        for file_path in slurm_files:
            result = self.parse_slurm_file(file_path)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("❌ No valid results found in SLURM files")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Add model names based on ablated components
        df['model_name'] = df['ablated_component'].apply(self.generate_model_name)
        
        # Calculate privacy-utility score if possible
        if 'accuracy' in df.columns and 'identity_accuracy' in df.columns:
            df['privacy_utility_score'] = df['accuracy'] - df['identity_accuracy']
        
        print(f"✅ Successfully parsed {len(df)} results")
        return df
    
    def generate_model_name(self, ablated_component):
        """Generate descriptive model name from ablated component."""
        name_mapping = {
            'bone_length': 'No Bone Length Loss',
            'foot_contact': 'No Foot Contact Loss',
            'joint_limit': 'No Joint Limit Loss',
            'velocity': 'No Velocity Loss',
            'end_effector': 'No End Effector Loss',
            'smoothing': 'No Smoothing Loss',
            'full_model': 'Full Model (Baseline)',
            'Unknown': 'Unknown Configuration'
        }
        return name_mapping.get(ablated_component, f'No {ablated_component.title()} Loss')
    
    def save_results(self, df):
        """Save parsed results to files."""
        if df is None or df.empty:
            print("❌ No data to save")
            return
        
        # Save as CSV
        csv_path = self.output_dir / "slurm_parsed_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.output_dir / "slurm_parsed_results.json"
        df.to_json(json_path, orient='records', indent=2)
        
        # Create summary
        summary = {
            'total_models': len(df),
            'parsing_date': datetime.now().isoformat(),
            'metrics_found': list(df.select_dtypes(include=['number']).columns),
            'ablated_components': df['ablated_component'].unique().tolist()
        }
        
        summary_path = self.output_dir / "slurm_parsing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to:")
        print(f"  📄 {csv_path}")
        print(f"  📄 {json_path}")
        print(f"  📄 {summary_path}")
        
        return csv_path, json_path, summary_path
    
    def run_parsing(self):
        """Run complete SLURM parsing process."""
        print("🔍 Starting SLURM ablation results parsing...")
        
        df = self.parse_all_slurm_files()
        if df is not None:
            self.save_results(df)
            
            # Print summary
            print(f"\n📊 Parsing Summary:")
            print(f"  📈 Total models: {len(df)}")
            print(f"  🔬 Ablated components: {', '.join(df['ablated_component'].unique())}")
            if 'accuracy' in df.columns:
                print(f"  🎯 Accuracy range: {df['accuracy'].min():.1f}% - {df['accuracy'].max():.1f}%")
            
        return df


def main():
    """Main execution function."""
    print("🔍 SLURM Ablation Results Parser")
    print("=" * 40)
    
    parser = SlurmAblationParser()
    results = parser.run_parsing()
    
    if results is not None:
        print("\n✅ Parsing complete!")
    else:
        print("\n❌ Parsing failed!")
    
    return results


if __name__ == "__main__":
    main()
