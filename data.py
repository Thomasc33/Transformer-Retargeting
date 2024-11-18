import random
import pickle
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, Manager

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
}

def load_data(dataset, T=64):
    assert dataset in datasets, f'Dataset {dataset} not found'
    
    # Load the data file
    with open(datasets[dataset]['path'], 'rb') as f:
        data = pickle.load(f)
    
    processed_data = {}
    for k, v in data.items():
        # Keep only the relevant joint and channel data if max_actors is 1
        if datasets[dataset]['max_actors'] == 1:
            v = v[:, :datasets[dataset]['joints'] * datasets[dataset]['channels']]
        
        # Remove zero frames along axis=1 (frame level)
        non_zero_frames = v[~np.all(v == 0, axis=1)]
        
        # Adjust the sequence length to T
        if len(non_zero_frames) < T:
            # If shorter than T, repeat the last frame until reaching T frames
            last_frame = non_zero_frames[-1:] if len(non_zero_frames) > 0 else np.zeros((1, non_zero_frames.shape[1]))
            num_repeats = T - len(non_zero_frames)
            padded_sequence = np.vstack([non_zero_frames] + [last_frame] * num_repeats)
        else:
            # Clip to T frames
            padded_sequence = non_zero_frames[:T]
        
        # Store the processed sequence in the dictionary
        processed_data[k] = padded_sequence
    
    return processed_data



def parse_file_name(file_name):
    """Parses the filename into a dictionary of parts."""
    file_name = str(file_name)
    S = int(file_name[1:4])
    C = int(file_name[5:8])
    P = int(file_name[9:12])
    R = int(file_name[13:16])
    A = int(file_name[17:20])
    return {'S': S, 'C': C, 'P': P, 'R': R, 'A': A}

def organize_data(data, setting, dataset='ntu120', T=64):
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    assert dataset in datasets, f'Dataset {dataset} not found'
    train_cameras = datasets[dataset]['train_cameras']
    test_cameras = datasets[dataset]['test_cameras']
    train_actors = datasets[dataset]['train_actors']

    organized_data = defaultdict(list)
    for file_name in data.keys():
        parts = parse_file_name(file_name)
        organized_data[parts['C']].append((parts['P'], parts['A'], file_name))
        if setting == 'cs':
            if parts['P'] in train_actors:
                train_data[parts['C']].append((parts['P'], parts['A'], file_name))
            else:
                test_data[parts['C']].append((parts['P'], parts['A'], file_name))

    if setting == 'cv':
        for camera in train_cameras:
            train_data[camera].extend(organized_data[camera])

        for camera in test_cameras:
            test_data[camera].extend(organized_data[camera])

    return train_data, test_data

def sample_data(organized_data):
    # Pick a random camera C
    C = random.choice(list(organized_data.keys()))

    # Get all (P, A, fname) tuples for this C 
    pa_list = organized_data[C]

    # Build a mapping from actors to actions they have performed
    actor_to_actions = defaultdict(set)
    for p, a, fname in pa_list:
        actor_to_actions[p].add(a)

    # Find actors who have performed at least two actions
    actors_with_multiple_actions = [p for p, actions in actor_to_actions.items() if len(actions) >= 2]

    if len(actors_with_multiple_actions) < 2:
        # Not enough actors with multiple actions
        return None

    # Shuffle actors to randomize selection
    random.shuffle(actors_with_multiple_actions)
    for i in range(len(actors_with_multiple_actions)):
        p1 = actors_with_multiple_actions[i]
        for j in range(i + 1, len(actors_with_multiple_actions)):
            p2 = actors_with_multiple_actions[j]
            common_actions = actor_to_actions[p1].intersection(actor_to_actions[p2])
            if len(common_actions) >= 2:
                # Found two actors with at least two common actions
                a1, a2 = random.sample(common_actions, 2)
                # Now get file names for (p1,a1), (p1,a2), (p2,a1), (p2,a2)
                pa_to_fname = {}
                for p, a, fname in pa_list:
                    key = (p, a)
                    if key not in pa_to_fname:
                        pa_to_fname[key] = fname
                required_keys = [(p1, a1), (p1, a2), (p2, a1), (p2, a2)]
                if all(k in pa_to_fname for k in required_keys):
                    sampled_data = [
                        (p1, a1, pa_to_fname[(p1, a1)]),
                        (p1, a2, pa_to_fname[(p1, a2)]),
                        (p2, a1, pa_to_fname[(p2, a1)]),
                        (p2, a2, pa_to_fname[(p2, a2)]),
                    ]
                    return sampled_data
    # If we reach here, no suitable pair found
    return None

def gen_samples_single_threaded(samples, data):
    d = []
    seen = set()
    failed_attempts = 0
    max_failed_attempts = 10000
    while len(d) < samples and failed_attempts < max_failed_attempts:
        result = sample_data(data)
        if result is None:
            failed_attempts += 1
            continue
        # Create a unique key for the sample to avoid duplicates
        key = tuple(sorted((p, a) for p, a, _ in result))
        if key not in seen:
            seen.add(key)
            d.append(result)
        else:
            failed_attempts += 1
    if failed_attempts >= max_failed_attempts:
        print('Failed to sample enough data without duplicates')
    return d

def worker(args):
    data, shared_seen_dict, lock = args
    result = None
    failed_attempts = 0
    max_failed_attempts = 1000
    while result is None and failed_attempts < max_failed_attempts:
        result = sample_data(data)
        if result is None:
            failed_attempts += 1
            continue
        else:
            key = tuple(sorted((p, a) for p, a, _ in result))
            with lock:
                if key in shared_seen_dict:
                    failed_attempts += 1
                    result = None
                else:
                    shared_seen_dict[key] = None  # Value can be anything
                    return result
    return None

def gen_samples(samples, data):
    from multiprocessing import Pool, Manager

    manager = Manager()
    shared_seen_dict = manager.dict()
    lock = manager.Lock()
    pool = Pool(processes=32)  # Adjust the number of processes as per your node's CPU cores

    args = [(data, shared_seen_dict, lock) for _ in range(samples * 2)]  # Generate more to account for duplicates
    results = pool.map(worker, args)
    pool.close()
    pool.join()

    # Filter out None results and limit to the desired number of samples
    unique_results = [res for res in results if res is not None]
    if len(unique_results) < samples:
        print('Failed to generate enough unique samples')
    return unique_results[:samples]


def process_trainining_data(X, setting='cs', dataset='ntu120'):
    if dataset not in datasets:
        raise ValueError(f'Dataset {dataset} not found')
    
    x_train, x_test, y_train, y_test = [], [], [], []
    for file in X:
        file_info = parse_file_name(file)
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

    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)

def process_mlm(data, setting, dataset='ntu120', T=64):
    train_data = []
    test_data = []

    assert dataset in datasets, f'Dataset {dataset} not found'
    train_cameras = datasets[dataset]['train_cameras']
    test_cameras = datasets[dataset]['test_cameras']
    train_actors = datasets[dataset]['train_actors']

    organized_data = defaultdict(list)
    for file_name in data.keys():
        parts = parse_file_name(file_name)
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
    def __init__(self, sampled_data, X):
        self.X = X  # The dictionary with skeleton sequences
        self.sampled_data = sampled_data  # List of sampled data
        # Extract actors and actions for fast retrieval
        self.actors = np.array([[sample[0][0], sample[2][0]] for sample in sampled_data], dtype=float)
        self.actions = np.array([[sample[0][1], sample[2][1]] for sample in sampled_data], dtype=float)

    def __getitem__(self, index):
        sample = self.sampled_data[index]
        # Load the skeleton sequences from self.X using the file names
        x1 = self.X[sample[0][2]]  # P1, A1
        x2 = self.X[sample[3][2]]  # P2, A2
        y1 = self.X[sample[1][2]]  # P1, A2
        y2 = self.X[sample[2][2]]  # P2, A1

        return (
            x1,
            x2,
            y1,
            y2,
            self.actors[index],
            self.actions[index]
        )

    def __len__(self):
        return len(self.sampled_data)
    
class PT_Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
    
class Masked_AE_Data(Dataset):
    def __init__(self, X, frame_masking_ratio=0.5, joint_masking_ratio=0.5):
        self.X = X
        self.frame_masking_ratio = frame_masking_ratio
        self.joint_masking_ratio = joint_masking_ratio

    def __getitem__(self, index):
        x = self.X[index]
        # (frames, joints * 3)
        frames, joints_dim = x.shape
        joints = joints_dim // 3  # Assume joints * 3 for XYZ coordinates

        # Masking frames
        num_frames_to_mask = int(self.frame_masking_ratio * frames)
        frame_indices = np.arange(frames)
        masked_frame_indices = np.random.choice(frame_indices, size=num_frames_to_mask, replace=False)
        x[masked_frame_indices, :] = 0  # Set masked frames to zero

        # Masking joints
        num_joints_to_mask = int(self.joint_masking_ratio * joints)
        joint_indices = np.arange(joints)
        masked_joint_indices = np.random.choice(joint_indices, size=num_joints_to_mask, replace=False)
        
        # Set masked joints to zero for all frames
        for joint_idx in masked_joint_indices:
            x[:, joint_idx * 3:(joint_idx + 1) * 3] = 0

        return x
    
    def __len__(self):
        return len(self.X)

def get_cross_data(X, dataset, setting, batch_size=32, return_loader=False, train_samples=50000, test_samples=5000):
    organized_data_train, organized_data_test = organize_data(X, setting, dataset)
    train_data = gen_samples(train_samples, organized_data_train)
    val_data = gen_samples(test_samples, organized_data_test)
    train_dataset = Cross_Data(train_data, X)
    val_dataset = Cross_Data(val_data, X)
    if return_loader:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl
    return train_dataset, val_dataset
