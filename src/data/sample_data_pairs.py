import logging
import sys
import torch
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from data import (
    load_data, parse_file_name, datasets, sample_frames,
    Cross_Data, _transform, build_group_key
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'sample_data_pairs.log')
    ]
)
logger = logging.getLogger(__name__)

def find_actor_pairs_with_shared_actions(data, dataset):
    logger.info("Finding all actor pairs with shared actions...")

    # Create mappings to organize data
    actor_to_actions = defaultdict(set)  # actor -> set of actions
    actor_to_setups = defaultdict(set)   # actor -> set of setups
    action_setup_camera_to_files = defaultdict(list)  # (actor, action, setup, camera) -> list of filenames

    # First pass: collect all actions, setups, and cameras for each actor
    for file_name in data.keys():
        parts = parse_file_name(file_name, dataset=dataset)

        # Only process files that match our repeat value (R001)
        if 'R' in parts and parts['R'] != 1:
            continue

        # Record this action for this actor
        actor_to_actions[parts['P']].add(parts['A'])

        # Record this setup for this actor
        if 'S' in parts:  # NTU datasets have setup info
            actor_to_setups[parts['P']].add(parts['S'])

            # Store the filename under (actor, action, setup, camera)
            # This allows us to easily find files for specific camera views
            action_setup_camera_to_files[(parts['P'], parts['A'], parts['S'], parts['C'])].append(file_name)
        else:
            # For datasets without setup info
            action_setup_camera_to_files[(parts['P'], parts['A'], None, parts['C'])].append(file_name)

    # Create a more convenient mapping for the generate_paired_samples function
    # (actor, action, setup) -> list of filenames
    action_setup_to_files = defaultdict(list)
    for (actor, action, setup, _), files in action_setup_camera_to_files.items():
        action_setup_to_files[(actor, action, setup)].extend(files)

    # Second pass: find pairs of actors with shared actions and setups
    actor_pairs = {}
    all_actors = sorted(actor_to_actions.keys())

    for i in range(len(all_actors)):
        actor1 = all_actors[i]
        for j in range(i + 1, len(all_actors)):
            actor2 = all_actors[j]

            # Find shared actions between these two actors
            shared_actions = actor_to_actions[actor1].intersection(actor_to_actions[actor2])

            # Skip if there are fewer than 2 shared actions (we need at least 2 for cross-action samples)
            if len(shared_actions) < 2:
                continue

            # For NTU datasets, also find shared setups
            if dataset in ['ntu', 'ntu120']:
                shared_setups = actor_to_setups[actor1].intersection(actor_to_setups[actor2])
                if len(shared_setups) == 0:
                    continue  # No shared setups, skip this pair

                # Store the pair with their shared actions and setups
                actor_pairs[(actor1, actor2)] = {
                    'shared_actions': shared_actions,
                    'shared_setups': shared_setups
                }
            else:
                # For datasets without setup info, just store shared actions
                actor_pairs[(actor1, actor2)] = {
                    'shared_actions': shared_actions,
                    'shared_setups': {None}  # Use None as a placeholder
                }

    logger.info(f"Found {len(actor_pairs)} actor pairs with shared actions")
    return actor_pairs, action_setup_to_files

def generate_paired_samples(actor_pairs, action_setup_to_files, dataset):
    logger.info("Generating paired samples...")

    # Get dataset configuration
    train_cameras = datasets[dataset]['train_cameras']

    train_samples = []
    test_samples = []

    # Process each actor pair
    for (actor1, actor2), pair_info in actor_pairs.items():
        shared_actions = list(pair_info['shared_actions'])
        shared_setups = list(pair_info['shared_setups'])

        # Skip if we don't have at least 2 shared actions
        if len(shared_actions) < 2:
            continue

        # For each setup and each pair of actions, create samples
        for setup in shared_setups:
            for i in range(len(shared_actions)):
                action1 = shared_actions[i]
                for j in range(i + 1, len(shared_actions)):
                    action2 = shared_actions[j]

                    # Try to find files for all combinations
                    keys = [
                        (actor1, action1, setup),
                        (actor1, action2, setup),
                        (actor2, action1, setup),
                        (actor2, action2, setup)
                    ]

                    # Check if all keys exist in our mapping
                    if not all(key in action_setup_to_files for key in keys):
                        continue

                    # For each key, get all available files (not just the first one)
                    # This allows us to create samples from different camera views
                    all_files_by_key = []
                    for key in keys:
                        if action_setup_to_files[key]:
                            all_files_by_key.append(action_setup_to_files[key])
                        else:
                            break

                    if len(all_files_by_key) != 4:
                        continue  # Couldn't find files for all keys

                    # Generate all possible combinations of files
                    # This will create multiple samples from the same actor-action pairs
                    # if they have recordings from different cameras
                    for file0 in all_files_by_key[0]:
                        # Get camera of the first file to determine train/test split
                        parts0 = parse_file_name(file0, dataset=dataset)
                        camera0 = parts0['C']

                        for file1 in all_files_by_key[1]:
                            # Check if the second file has the same camera as the first
                            parts1 = parse_file_name(file1, dataset=dataset)
                            if parts1['C'] != camera0:
                                continue

                            for file2 in all_files_by_key[2]:
                                # Check if the third file has the same camera as the first
                                parts2 = parse_file_name(file2, dataset=dataset)
                                if parts2['C'] != camera0:
                                    continue

                                for file3 in all_files_by_key[3]:
                                    # Check if the fourth file has the same camera as the first
                                    parts3 = parse_file_name(file3, dataset=dataset)
                                    if parts3['C'] != camera0:
                                        continue

                                    # Create the sample tuple
                                    sample = [
                                        (actor1, action1, file0),
                                        (actor1, action2, file1),
                                        (actor2, action1, file2),
                                        (actor2, action2, file3)
                                    ]

                                    # Determine if this is a training or testing sample
                                    # based on the camera ID
                                    if camera0 in train_cameras:
                                        train_samples.append(sample)
                                    else:
                                        test_samples.append(sample)

    logger.info(f"Generated {len(train_samples)} training samples and {len(test_samples)} testing samples")
    return train_samples, test_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ntu', help='Dataset to use (ntu, ntu120, etri)')
    parser.add_argument('--no-save', action='store_true', help='Do not save the generated samples')

    args = parser.parse_args()

    dataset = args.dataset
    save_samples = not args.no_save
    T = 64  # Number of frames to sample

    logger.info(f"Starting comprehensive data sampling for {dataset} dataset with cross-view setting")

    paired_file_path = f'data/{dataset}_cv_paired_comprehensive.pt'

    # Load data
    logger.info(f"Loading {dataset} dataset...")
    X = load_data(dataset, T)
    logger.info(f"Loaded {len(X)} sequences from {dataset} dataset")

    # Find all actor pairs with shared actions
    actor_pairs, action_setup_to_files = find_actor_pairs_with_shared_actions(X, dataset)

    # Generate paired samples
    train_samples, test_samples = generate_paired_samples(
        actor_pairs, action_setup_to_files, dataset
    )

    # Create dataset objects
    logger.info("Creating dataset objects...")
    train_dataset = Cross_Data(train_samples, X, seg=T, augment=True, theta=0.5)
    test_dataset = Cross_Data(test_samples, X, seg=T, augment=False)

    if save_samples:
        logger.info(f"Saving paired data to {paired_file_path}")
        torch.save({'train': train_dataset, 'test': test_dataset}, paired_file_path)
        logger.info('Paired data saved successfully')

    logger.info(f"Final dataset sizes: {len(train_dataset)} training samples, {len(test_dataset)} testing samples")

    return train_dataset, test_dataset

if __name__ == "__main__":
    main()
