import random
import pickle
import logging
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, Manager

# Configure logging
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Data Augmentation Functions
#------------------------------------------------------------------------------

def _rot(rot):
    """
    Create rotation matrices from rotation angles
    """
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros), dim=-1)
    rx2 = torch.stack((zeros, cos_r[:, :, 0:1], sin_r[:, :, 0:1]), dim=-1)
    rx3 = torch.stack((zeros, -sin_r[:, :, 0:1], cos_r[:, :, 0:1]), dim=-1)
    rx = torch.cat((r1, rx2, rx3), dim=2)

    ry1 = torch.stack((cos_r[:, :, 1:2], zeros, -sin_r[:, :, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, :, 1:2], zeros, cos_r[:, :, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=2)

    rz1 = torch.stack((cos_r[:, :, 2:3], sin_r[:, :, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, :, 2:3], cos_r[:, :, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=2)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _transform(x, theta):
    """
    Apply random rotations for data augmentation
    """
    x = x.contiguous().view(x.size()[:2] + (-1, 3))
    rot = x.new(x.size()[0], 3).uniform_(-theta, theta)
    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def sample_frames(sequence, seg):
    """
    Sample frames from a sequence using the same technique as in eval_loader.py

    Args:
        sequence: numpy array of shape (frames, features)
        seg: number of frames to sample

    Returns:
        numpy array of shape (seg, features)
    """
    # Remove zero frames (frames where all values are zero)
    non_zero_mask = ~np.all(sequence == 0, axis=1)
    non_zero_frames = sequence[non_zero_mask]

    # If sequence is shorter than seg frames, repeat the last frame
    if len(non_zero_frames) < seg:
        if len(non_zero_frames) > 0:
            # Get the last frame and repeat it
            last_frame = non_zero_frames[-1:]
            num_repeats = seg - len(non_zero_frames)
            repeated_frames = np.repeat(last_frame, num_repeats, axis=0)
            processed_seq = np.concatenate([non_zero_frames, repeated_frames], axis=0)
        else:
            # Handle edge case: if non_zero_frames is empty
            processed_seq = np.zeros((seg, sequence.shape[1]), dtype=np.float32)
    else:
        # Sample frames with equal spacing as in the original SGN implementation
        num_frames = len(non_zero_frames)
        ave_duration = num_frames // seg

        # Sample frames at regular intervals with small random offset
        offsets = np.multiply(list(range(seg)), ave_duration) + np.random.randint(ave_duration, size=seg)

        # Ensure we don't go out of bounds
        offsets = np.clip(offsets, 0, num_frames-1)
        processed_seq = non_zero_frames[offsets]

    return processed_seq

#------------------------------------------------------------------------------
# Dataset Configurations
#------------------------------------------------------------------------------

datasets = {
    'ntu120': {
        'path': 'data/ntu120/ntu120.pkl',
        'max_actors': 1,
        'joints': 25,
        'channels': 3,
        'train_cameras': [2, 3],
        'test_cameras': [1],
        'train_actors': [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103],
        'num_class': 120,
        'num_actor': 106,
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
    },
    'ntu': {
        'path': 'data/ntu/ntu.pkl',
        'max_actors': 1,
        'joints': 25,
        'channels': 3,
        'train_cameras': [2, 3],
        'test_cameras': [1],
        'train_actors': [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38],
        'num_class': 60,
        'num_actor': 40,
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
    },
    'etri': {
        'path': 'data/etri/etri.pkl',
        'max_actors': 1,
        'joints': 25,
        'channels': 3,
        'train_cameras': [2, 3, 5, 7],
        'test_cameras': [1, 4, 6, 8],
        'train_actors': [84, 24, 51, 12, 66, 58, 1, 56, 54, 62, 22, 42, 30, 64, 41, 69, 3, 80, 21, 35, 90, 89, 63, 46, 32, 2, 26, 72, 50, 91, 16, 57, 36, 71, 31, 59, 78, 53, 9, 27, 7, 95, 4, 83, 65, 48, 75, 5, 44, 100],
        'num_class': 55,
        'num_actor': 100,
        'graph': 'graph.ntu_rgb_d.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
    }
}

#------------------------------------------------------------------------------
# Data Loading Functions
#------------------------------------------------------------------------------

def load_data(dataset, T=64):
    """
    Loads the raw data from the dataset pickle, truncates/pads each sequence to length T.

    Args:
        dataset: str - Name of the dataset ('ntu', 'ntu120', 'etri')
        T: int - Target sequence length

    Returns:
        dict - Mapping of filename to processed skeleton data (frames x joints*channels)
    """
    assert dataset in datasets, f'Dataset {dataset} not found'

    # Load the data file
    with open(datasets[dataset]['path'], 'rb') as f:
        data = pickle.load(f)

    processed_data = {}
    for k, v in data.items():
        # Remove 2 actor actions (50-60 and 106-120)
        if dataset in ['ntu', 'ntu120']:
            remove = set([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        106, 107, 108, 109, 110, 111, 112, 113, 114,
                        115, 116, 117, 118, 119, 120])
            action = parse_file_name(k, dataset)['A']
            if action in remove:
                continue

        # Keep only the relevant joint and channel data if max_actors is 1
        if datasets[dataset]['max_actors'] == 1:
            v = v[:, :datasets[dataset]['joints'] * datasets[dataset]['channels']]

        # Remove zero frames (frames that are all zeros)
        non_zero_frames = v[~np.all(v == 0, axis=1)]

        # Adjust the sequence length to T
        if len(non_zero_frames) < T:
            # If shorter than T, repeat the last frame until reaching T frames
            if len(non_zero_frames) > 0:
                last_frame = non_zero_frames[-1:]
            else:
                # Handle edge-case: if non_zero_frames is empty
                last_frame = np.zeros((1, v.shape[1]))
            num_repeats = T - len(non_zero_frames)
            padded_sequence = np.vstack([non_zero_frames] + [last_frame] * num_repeats)
        else:
            # Clip to T frames
            padded_sequence = non_zero_frames[:T]

        processed_data[k] = padded_sequence

    return processed_data

def get_num_classes(dataset, type='ar'):
    """
    Returns the number of classes for the specified dataset and type.

    Args:
        dataset: str - Name of the dataset
        type: str - 'ar' for action recognition, 'ri' for re-identification

    Returns:
        int - Number of classes
    """
    assert dataset in datasets, f'Dataset {dataset} not found'

    if type == 'ar':
        return datasets[dataset]['num_class']
    elif type == 'ri':
        return datasets[dataset]['num_actor']
    else:
        raise ValueError(f'Unknown type {type}. Use "ar" or "ri".')

#------------------------------------------------------------------------------
# Filename Parsing Functions
#------------------------------------------------------------------------------

def parse_file_name(file_name, dataset='ntu'):
    """
    Parses the filename into a dictionary of parts.

    Args:
        file_name: str - Filename to parse
        dataset: str - Dataset name to determine parsing format

    Returns:
        dict - Parsed components of the filename
    """
    file_name = str(file_name)
    if dataset in ['ntu', 'ntu120']:
        S = int(file_name[1:4])  # Setup
        C = int(file_name[5:8])  # Camera
        P = int(file_name[9:12])  # Person/Actor
        R = int(file_name[13:16])  # Replication
        A = int(file_name[17:20])  # Action
        return {'S': S, 'C': C, 'P': P, 'R': R, 'A': A}
    elif dataset == 'etri':
        A = int(file_name[1:4])  # Action
        P = int(file_name[5:8])  # Person/Actor
        G = int(file_name[9:12])  # Group (ignored)
        C = int(file_name[13:16])  # Camera
        return {'A': A, 'P': P, 'G': G, 'C': C}


def build_group_key(parts, dataset):
    """
    Builds the grouping key for organizing data.

    Args:
        parts: dict - Parsed filename parts
        dataset: str - Dataset name

    Returns:
        tuple or int - Grouping key (S,C) for NTU datasets, C for ETRI
    """
    if dataset in ['ntu', 'ntu120']:
        return (parts['S'], parts['C'])
    elif dataset == 'etri':
        return parts['C']

#------------------------------------------------------------------------------
# Data Organization Functions
#------------------------------------------------------------------------------

def organize_data(data, setting, dataset='ntu120', T=64):
    """
    Organizes data into train and test sets based on dataset and setting.

    Args:
        data: dict - Raw data mapping filenames to sequences
        setting: str - 'cs' for cross-subject, 'cv' for cross-view
        dataset: str - Dataset name
        T: int - Target sequence length

    Returns:
        tuple - (train_data, test_data) dictionaries organized by group keys
    """
    assert dataset in datasets, f'Dataset {dataset} not found'
    train_cameras = datasets[dataset]['train_cameras']
    test_cameras = datasets[dataset]['test_cameras']
    train_actors = datasets[dataset]['train_actors']

    # Step 1: Create a structure to hold pa_list for each group_key
    train_data_raw = defaultdict(list)
    test_data_raw = defaultdict(list)

    # We'll also keep a global aggregator in case we need for 'cv'
    organized_data_raw = defaultdict(list)

    for file_name in data.keys():
        parts = parse_file_name(file_name, dataset=dataset)
        group_key = build_group_key(parts, dataset)

        # Keep track globally
        organized_data_raw[group_key].append((parts['P'], parts['A'], file_name))

        if setting == 'cs':
            # cross-subject
            if parts['P'] in train_actors:
                train_data_raw[group_key].append((parts['P'], parts['A'], file_name))
            else:
                test_data_raw[group_key].append((parts['P'], parts['A'], file_name))

    if setting == 'cv':
        # cross-view
        # For NTU / ETRI, we look at cameras
        # but note that group_key is (S,C) for NTU or just C for ETRI
        # We'll check the 'C' portion for train/test
        for gk in organized_data_raw:
            # gk is either (S, C) or just C
            if isinstance(gk, tuple):
                # e.g. (S, C)
                _, c = gk
            else:
                c = gk
            if c in train_cameras:
                train_data_raw[gk].extend(organized_data_raw[gk])
            elif c in test_cameras:
                test_data_raw[gk].extend(organized_data_raw[gk])

    # Step 2: Convert these raw lists into summary dictionaries
    train_data = {}
    for gk, pa_list in train_data_raw.items():
        if len(pa_list) == 0:
            continue
        train_data[gk] = build_summary(pa_list)

    test_data = {}
    for gk, pa_list in test_data_raw.items():
        if len(pa_list) == 0:
            continue
        test_data[gk] = build_summary(pa_list)

    return train_data, test_data

def build_summary(pa_list):
    """
    Convert raw (person, action, filename) lists into organized dictionaries.

    Args:
        pa_list: list - List of (person, action, filename) tuples

    Returns:
        dict - Summary containing actor_to_actions, pa_map, and actors_with_2plus
    """
    actor_to_actions = defaultdict(set)
    pa_map = defaultdict(list)

    for p, a, fname in pa_list:
        actor_to_actions[p].add(a)
        # Keep a list of possible filenames in case duplicates exist
        pa_map[(p, a)].append(fname)

    # Find all actors that have at least 2 distinct actions
    actors_with_2plus = [p for p, actions in actor_to_actions.items() if len(actions) >= 2]

    summary = {
        'actor_to_actions': actor_to_actions,
        'pa_map': pa_map,
        'actors_with_2plus': actors_with_2plus
    }
    return summary

#------------------------------------------------------------------------------
# Data Sampling Functions
#------------------------------------------------------------------------------

def sample_data(organized_dict):
    """
    Randomly samples data to find pairs of different actors with common actions.

    Args:
        organized_dict: dict - Organized data dictionary

    Returns:
        list or None - List of (p1, a1, fname), (p1, a2, fname), (p2, a1, fname), (p2, a2, fname)
                      or None if no valid sample found
    """
    if not organized_dict:
        return None

    # Pick a random group_key
    group_key = random.choice(list(organized_dict.keys()))
    summary = organized_dict[group_key]

    actors_2plus = summary['actors_with_2plus']
    if len(actors_2plus) < 2:
        return None

    # Shuffle to randomize selection
    random.shuffle(actors_2plus)

    actor_to_actions = summary['actor_to_actions']
    pa_map = summary['pa_map']

    # We'll attempt to find two actors with 2 or more intersecting actions
    for i in range(len(actors_2plus)):
        p1 = actors_2plus[i]
        for j in range(i + 1, len(actors_2plus)):
            p2 = actors_2plus[j]
            common_actions = actor_to_actions[p1].intersection(actor_to_actions[p2])
            if len(common_actions) >= 2:
                # Found at least 2 common actions
                a1, a2 = random.sample(common_actions, 2)
                # We need the filenames for (p,a1), (p,a2) for both p1 and p2
                # It's possible there could be multiple filenames for each (p,a)
                # We'll just pick one at random from pa_map
                needed_keys = [(p1, a1), (p1, a2), (p2, a1), (p2, a2)]
                # Check if all exist
                for nk in needed_keys:
                    if nk not in pa_map or len(pa_map[nk]) == 0:
                        break
                else:
                    # We can sample one fname from each key
                    p1_a1_fname = random.choice(pa_map[(p1, a1)])
                    p1_a2_fname = random.choice(pa_map[(p1, a2)])
                    p2_a1_fname = random.choice(pa_map[(p2, a1)])
                    p2_a2_fname = random.choice(pa_map[(p2, a2)])
                    sample_result = [
                        (p1, a1, p1_a1_fname),
                        (p1, a2, p1_a2_fname),
                        (p2, a1, p2_a1_fname),
                        (p2, a2, p2_a2_fname)
                    ]
                    logger.info(f"Found sample: actors [{p1}, {p2}], actions [{a1}, {a2}], group {group_key}")
                    return sample_result
    return None


def gen_samples_single_threaded(samples, organized_dict):
    """
    Single-threaded sampling of the needed number of pairs,
    avoiding duplicates by storing them in a local set.
    """
    results = []
    seen = set()
    failed_attempts = 0
    max_failed_attempts = 10000
    last_log_count = 0
    log_interval = max(1, samples // 10)  # Log progress at 10% intervals

    logger.info(f"Starting single-threaded sampling to find {samples} samples")

    while len(results) < samples and failed_attempts < max_failed_attempts:
        result = sample_data(organized_dict)
        if result is None:
            failed_attempts += 1
            continue

        # Create a canonical key (sorted by p,a) to avoid duplicates
        key = tuple(sorted((p, a) for p, a, _ in result))
        if key not in seen:
            seen.add(key)
            results.append(result)

            # Log progress at intervals
            if len(results) - last_log_count >= log_interval:
                last_log_count = len(results)
                progress_pct = (len(results) / samples) * 100
                logger.info(f"Sampling progress: {len(results)}/{samples} samples ({progress_pct:.1f}%)")
        else:
            failed_attempts += 1

    if failed_attempts >= max_failed_attempts:
        logger.warning('Failed to sample enough data without duplicates (single-thread).')
    else:
        logger.info(f"Successfully sampled {len(results)} samples (single-threaded)")

    return results


def worker(args):
    """
    Worker function for multiprocessing.
    args = (organized_dict, shared_seen_dict, lock)
    """
    organized_dict, shared_seen_dict, lock = args
    result = None
    failed_attempts = 0
    max_failed_attempts = 1000

    while result is None and failed_attempts < max_failed_attempts:
        result = sample_data(organized_dict)
        if result is None:
            failed_attempts += 1
            continue
        else:
            key = tuple(sorted((p, a) for p, a, _ in result))
            with lock:
                if key in shared_seen_dict:
                    # Already seen, try again
                    failed_attempts += 1
                    result = None
                else:
                    # Mark it as seen
                    shared_seen_dict[key] = None
                    return result
    return None


def gen_samples(samples, organized_dict, threads=1):
    """
    Multithreaded sampling using Pool and a Manager dict to avoid duplicates.
    We create more jobs than 'samples' to allow for duplicates that get filtered out.
    """
    if threads <= 1:
        # Just do single-threaded if only 1 thread is requested
        return gen_samples_single_threaded(samples, organized_dict)

    logger.info(f"Starting multi-threaded sampling with {threads} threads to find {samples} samples")

    from multiprocessing import Pool, Manager
    manager = Manager()
    shared_seen_dict = manager.dict()
    lock = manager.Lock()

    pool = Pool(processes=threads)
    # We'll generate 2x samples worth of tasks to handle duplicates
    args = [(organized_dict, shared_seen_dict, lock) for _ in range(samples * 2)]
    results = pool.map(worker, args)
    pool.close()
    pool.join()

    # Filter out None results
    unique_results = [res for res in results if res is not None]
    found_count = len(unique_results)

    if found_count < samples:
        logger.warning(f"Found only {found_count}/{samples} unique samples (multi-thread).")
    else:
        logger.info(f"Successfully sampled {min(found_count, samples)}/{samples} samples (multi-threaded)")

    return unique_results[:samples]


def process_trainining_data(X, setting='cs', dataset='ntu120'):
    """
    Converts the raw skeleton dictionary X into train/test arrays (x_train, y_train, x_test, y_test).
    For cross-subject (cs), uses train_actors. For cross-view (cv), uses train_cameras.
    """
    if dataset not in datasets:
        raise ValueError(f'Dataset {dataset} not found')

    x_train, x_test, y_train, y_test = [], [], [], []

    for file in X:
        file_info = parse_file_name(file, dataset=dataset)
        if setting == 'cs':
            if file_info['P'] in datasets[dataset]['train_actors']:
                x_train.append(X[file])
                y_train.append([file_info['A']-1, file_info['P']-1])
            else:
                x_test.append(X[file])
                y_test.append([file_info['A']-1, file_info['P']-1])
        elif setting == 'cv':
            if file_info['C'] in datasets[dataset]['train_cameras']:
                x_train.append(X[file])
                y_train.append([file_info['A']-1, file_info['P']-1])
            else:
                x_test.append(X[file])
                y_test.append([file_info['A']-1, file_info['P']-1])

    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )


def process_mlm(data, setting, dataset='ntu120', T=64):
    """
    Similar to organize_data but returns train/test file lists for Masked-Language-Model style training.
    """
    train_data = []
    test_data = []

    assert dataset in datasets, f'Dataset {dataset} not found'
    train_cameras = datasets[dataset]['train_cameras']
    test_cameras = datasets[dataset]['test_cameras']
    train_actors = datasets[dataset]['train_actors']

    organized_data = defaultdict(list)
    for file_name in data.keys():
        parts = parse_file_name(file_name, dataset=dataset)
        organized_data[parts['C']].append(file_name)
        if setting == 'cs':
            if parts['P'] in train_actors:
                train_data.append(file_name)
            else:
                test_data.append(file_name)

    if setting == 'cv':
        for camera in train_cameras:
            train_data.extend(organized_data[camera])
        for camera in test_cameras:
            test_data.extend(organized_data[camera])

    return train_data, test_data


class Cross_Data(Dataset):
    """
    For a list of 4-tuples:
      [
        (p1, a1, fname1),
        (p1, a2, fname2),
        (p2, a1, fname3),
        (p2, a2, fname4),
      ]
    We'll load x1 from fname1, x2 from fname4, y1 from fname2, y2 from fname3,
    and also store actor/action labels as needed.
    """
    def __init__(self, sampled_data, X, seg=64, augment=True, theta=0.3):
        self.X = X  # The dictionary: fname -> skeleton array
        self.sampled_data = sampled_data  # List of 4-tuples
        self.seg = seg  # Number of frames to sample
        self.augment = augment  # Whether to apply data augmentation
        self.theta = theta  # Rotation angle for augmentation

        self.actors = np.array([[sample[0][0], sample[2][0]] for sample in sampled_data], dtype=float)
        self.actions = np.array([[sample[0][1], sample[1][1]] for sample in sampled_data], dtype=float)

    def __getitem__(self, index):
        sample = self.sampled_data[index]
        # 0: (p1, a1, fname)
        # 1: (p1, a2, fname)
        # 2: (p2, a1, fname)
        # 3: (p2, a2, fname)

        # Get raw sequences
        x1_raw = self.X[sample[0][2]]  # P1, A1
        x2_raw = self.X[sample[3][2]]  # P2, A2
        y1_raw = self.X[sample[1][2]]  # P1, A2
        y2_raw = self.X[sample[2][2]]  # P2, A1

        # Apply frame sampling
        x1 = sample_frames(x1_raw, self.seg)
        x2 = sample_frames(x2_raw, self.seg)
        y1 = sample_frames(y1_raw, self.seg)
        y2 = sample_frames(y2_raw, self.seg)

        # Convert to torch tensors
        x1 = torch.from_numpy(x1).float()
        x2 = torch.from_numpy(x2).float()
        y1 = torch.from_numpy(y1).float()
        y2 = torch.from_numpy(y2).float()

        # Apply rotation augmentation if enabled
        if self.augment:
            x1 = _transform(x1.unsqueeze(0), self.theta).squeeze(0)
            x2 = _transform(x2.unsqueeze(0), self.theta).squeeze(0)
            y1 = _transform(y1.unsqueeze(0), self.theta).squeeze(0)
            y2 = _transform(y2.unsqueeze(0), self.theta).squeeze(0)

        return (
            x1,  # x1
            x2,  # x2
            y1,  # y1
            y2,  # y2
            self.actors[index],   # [p1, p2]
            self.actions[index],  # [a1, a2]
        )

    def __len__(self):
        return len(self.sampled_data)


class PT_Data(Dataset):
    """
    Simple supervised dataset from X, y arrays.
    """
    def __init__(self, X, y, seg=64, augment=True, theta=0.3):
        self.X = X
        self.y = y
        self.seg = seg  # Number of frames to sample
        self.augment = augment  # Whether to apply data augmentation
        self.theta = theta  # Rotation angle for augmentation

    def __getitem__(self, index):
        x_raw = self.X[index].numpy() if isinstance(self.X[index], torch.Tensor) else self.X[index]

        # Apply frame sampling
        x = sample_frames(x_raw, self.seg)

        # Convert to torch tensor
        x = torch.from_numpy(x).float()

        # Apply rotation augmentation if enabled
        if self.augment:
            x = _transform(x.unsqueeze(0), self.theta).squeeze(0)

        return x, self.y[index]

    def __len__(self):
        return len(self.X)


class Masked_AE_Data(Dataset):
    """
    Simple masked autoencoder style dataset.
    For each skeleton, randomly masks frames and joints.
    """
    def __init__(self, X, frame_masking_ratio=0.5, joint_masking_ratio=0.5, seg=64, augment=True, theta=0.3):
        self.X = X
        self.frame_masking_ratio = frame_masking_ratio
        self.joint_masking_ratio = joint_masking_ratio
        self.seg = seg  # Number of frames to sample
        self.augment = augment  # Whether to apply data augmentation
        self.theta = theta  # Rotation angle for augmentation

    def __getitem__(self, index):
        x_raw = self.X[index].numpy() if isinstance(self.X[index], torch.Tensor) else self.X[index].copy()

        # Apply frame sampling
        x = sample_frames(x_raw, self.seg)

        # Convert to torch tensor
        x = torch.from_numpy(x).float()

        # Apply rotation augmentation if enabled
        if self.augment:
            x = _transform(x.unsqueeze(0), self.theta).squeeze(0)

        # Convert back to numpy for masking
        x = x.numpy()
        frames, joints_dim = x.shape
        joints = joints_dim // 3  # If we have (joints * 3) columns

        # 1) Randomly mask frames
        num_frames_to_mask = int(self.frame_masking_ratio * frames)
        frame_indices = np.arange(frames)
        masked_frame_indices = np.random.choice(frame_indices, size=num_frames_to_mask, replace=False)
        x[masked_frame_indices, :] = 0

        # 2) Randomly mask joints
        num_joints_to_mask = int(self.joint_masking_ratio * joints)
        joint_indices = np.arange(joints)
        masked_joint_indices = np.random.choice(joint_indices, size=num_joints_to_mask, replace=False)
        for joint_idx in masked_joint_indices:
            x[:, joint_idx * 3:(joint_idx + 1) * 3] = 0

        # Convert back to torch tensor for return
        x = torch.from_numpy(x).float()
        return x

    def __len__(self):
        return len(self.X)


def get_cross_data(X, dataset, setting, batch_size=32, return_loader=False,
                   train_samples=50000, test_samples=5000, threads=1, seg=64,
                   augment=True, train_theta=0.3, val_theta=0.3):
    """
    Entry point to generate 'paired' cross data (two actors, two actions)
    for training and validation sets.

    Args:
        X: dict - Raw data mapping filenames to sequences
        dataset: str - Dataset name
        setting: str - 'cs' for cross-subject, 'cv' for cross-view
        batch_size: int - Batch size for DataLoader
        return_loader: bool - Whether to return DataLoader objects
        train_samples: int - Number of training samples to generate
        test_samples: int - Number of test samples to generate
        threads: int - Number of threads for sample generation
        seg: int - Number of frames to sample from each sequence
        augment: bool - Whether to apply data augmentation
        train_theta: float - Rotation angle for training data augmentation
        val_theta: float - Rotation angle for validation data augmentation

    Returns:
        tuple - (train_dataset, val_dataset) or (train_loader, val_loader)
    """
    logger.info(f"Starting data sampling for {dataset} dataset with {setting} setting")
    logger.info(f"Target: {train_samples} training samples, {test_samples} test samples")

    # 1) Organize data by group key
    logger.info("Organizing data by group key...")
    organized_data_train, organized_data_test = organize_data(X, setting, dataset)

    train_groups = len(organized_data_train)
    test_groups = len(organized_data_test)
    logger.info(f"Organized data into {train_groups} training groups and {test_groups} test groups")

    # 2) Generate samples either single- or multi-threaded
    logger.info("Generating training samples...")
    if threads == 1:
        train_data = gen_samples_single_threaded(train_samples, organized_data_train)
        logger.info("Generating test samples...")
        val_data = gen_samples_single_threaded(test_samples, organized_data_test)
    else:
        train_data = gen_samples(train_samples, organized_data_train, threads=threads)
        logger.info("Generating test samples...")
        val_data = gen_samples(test_samples, organized_data_test, threads=threads)

    # 3) Build the dataset objects with augmentation parameters
    logger.info("Building dataset objects...")
    train_dataset = Cross_Data(train_data, X, seg=seg, augment=augment, theta=train_theta)
    val_dataset = Cross_Data(val_data, X, seg=seg, augment=False)

    logger.info(f"Data sampling complete: {len(train_data)} training samples, {len(val_data)} test samples")

    # 4) Optionally wrap them in DataLoader
    if return_loader:
        logger.info("Creating data loaders...")
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl

    return train_dataset, val_dataset


# Optimized data loading functions (merged from data_optimized.py)
import multiprocessing as mp


def optimize_data_loading(train_dataset, val_dataset, batch_size, distributed=False, rank=0, world_size=1):
    """
    Create optimized data loaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        distributed: Whether using distributed training
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # OPTIMIZED: Aggressive worker count for maximum data loading performance
    if distributed:
        # For distributed training, use more workers per GPU
        num_workers = min(mp.cpu_count() // world_size, 16)  # INCREASED: Even more workers
    else:
        # For single GPU, use maximum available workers
        num_workers = min(mp.cpu_count(), 20)  # INCREASED: Maximum workers for single GPU

    if distributed and world_size > 1:
        # Use DistributedSampler for distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=8 if num_workers > 0 else 2,  # INCREASED: Maximum prefetching
            drop_last=True,  # Ensure consistent batch sizes for DDP
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=8 if num_workers > 0 else 2,  # INCREASED: Maximum prefetching
            drop_last=False,  # Keep all validation samples
        )
    else:
        # Single-process data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=False
        )

    return train_loader, val_loader


def estimate_memory_usage(dataset, batch_size, sample_size=10):
    """
    Estimate memory usage for a dataset and batch size.

    Args:
        dataset: Dataset to estimate for
        batch_size: Batch size
        sample_size: Number of samples to use for estimation

    Returns:
        Dictionary with memory usage estimates
    """
    sample_items = []
    for i in range(min(sample_size, len(dataset))):
        sample_items.append(dataset[i])

    # Calculate average item size
    total_size = 0
    for item in sample_items:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                if torch.is_tensor(sub_item):
                    total_size += sub_item.numel() * sub_item.element_size()
        elif torch.is_tensor(item):
            total_size += item.numel() * item.element_size()

    avg_item_size = total_size / len(sample_items)
    batch_memory = avg_item_size * batch_size

    return {
        'avg_item_size_mb': avg_item_size / (1024 * 1024),
        'batch_memory_mb': batch_memory / (1024 * 1024),
        'batch_memory_gb': batch_memory / (1024 * 1024 * 1024),
        'estimated_peak_memory_gb': batch_memory * 3 / (1024 * 1024 * 1024)  # Rough estimate including gradients
    }
