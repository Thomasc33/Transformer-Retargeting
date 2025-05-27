"""
Simple test script to demonstrate loading and using paired data.
"""
import torch

# Load paired data from cache file
paired_data = torch.load('data/ntu_cs_paired.pt')
paired_train = paired_data['train']
paired_test = paired_data['test']

# Print initial dataset sizes
print("Initial dataset sizes:")
print(f"Training samples: {len(paired_train)}")
print(f"Testing samples: {len(paired_test)}")
print(f"Training dataset type: {type(paired_train)}")
print(f"Testing dataset type: {type(paired_test)}")

# Reduce dataset size for testing
paired_train.sampled_data = paired_train.sampled_data[:300]
paired_test.sampled_data = paired_test.sampled_data[:300]

# Print modified dataset sizes
print("\nModified dataset sizes (limited to 300 samples each):")
print(f"Training samples: {len(paired_train)}")
print(f"Testing samples: {len(paired_test)}")