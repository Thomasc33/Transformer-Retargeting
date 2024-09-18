#!/usr/bin/env python3
"""
Generate paired cross-identity training data for DisentangledTMR.

Creates .pt files containing Cross_Data objects with 2x2 actor-action
quadruplets for training the disentangled motion retargeting model.

Usage:
    python scripts/generate_paired_data.py --dataset etri --setting cv \
        --train_samples 10000 --test_samples 2000 \
        --output data/etri/etri_cv_paired_10k.pt
"""
import os
import sys
import argparse
import logging
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.datasets import load_data, organize_data, gen_samples, Cross_Data

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paired cross-identity data")
    parser.add_argument('--dataset', required=True, choices=['ntu', 'ntu120', 'etri'])
    parser.add_argument('--setting', default='cv', choices=['cv', 'cs'])
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--test_samples', type=int, default=2000)
    parser.add_argument('--output', required=True, help='Output .pt path')
    args = parser.parse_args()

    logger.info(f"Dataset: {args.dataset}, Setting: {args.setting}")
    logger.info(f"Samples: {args.train_samples} train, {args.test_samples} test")

    # Load raw data
    data_dict = load_data(args.dataset)
    logger.info(f"Loaded {len(data_dict)} raw samples")

    # Organize by camera/subject split
    train_dict, test_dict = organize_data(data_dict, args.setting, args.dataset)
    logger.info(f"Train groups: {len(train_dict)}, Test groups: {len(test_dict)}")

    # Sample cross-identity quadruplets
    logger.info(f"Sampling {args.train_samples} training pairs...")
    train_samples = gen_samples(args.train_samples, train_dict)
    logger.info(f"Got {len(train_samples)} training samples")

    logger.info(f"Sampling {args.test_samples} test pairs...")
    test_samples = gen_samples(args.test_samples, test_dict)
    logger.info(f"Got {len(test_samples)} test samples")

    # Create Cross_Data objects
    train_dataset = Cross_Data(train_samples, data_dict, augment=True)
    test_dataset = Cross_Data(test_samples, data_dict, augment=False)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'train': train_dataset, 'test': test_dataset}, args.output)
    logger.info(f"Saved to {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
