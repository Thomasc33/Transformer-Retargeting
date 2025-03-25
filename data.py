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


def load_data(dataset, T=64):
    """
    Loads the raw data from the dataset pickle, truncates/pads each sequence to length T,
    and returns a dictionary: filename -> (frames x (joints*channels)).
    """
    assert dataset in datasets, f'Dataset {dataset} not found'
    
    # Load the data file
    with open(datasets[dataset]['path'], 'rb') as f:
        data = pickle.load(f)
    
    processed_data = {}
    for k, v in data.items():
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
    'ar' for action recognition, 'pt' for pose tracking (if applicable).
    """
    assert dataset in datasets, f'Dataset {dataset} not found'
    
    if type == 'ar':
        return datasets[dataset]['num_class']
    elif type == 'ri':
        return datasets[dataset]['num_actor']
    else:
        raise ValueError(f'Unknown type {type}. Use "ar" or "pt".')


def parse_file_name(file_name, dataset='ntu'):
    """
    Parses the filename into a dictionary of parts.
    For NTU/NTU120: S##, C##, P##, R##, A##.
    For ETRI: A###, P###, G###, C###.
    """
    file_name = str(file_name)
    if dataset in ['ntu', 'ntu120']:
        S = int(file_name[1:4])
        C = int(file_name[5:8])
        P = int(file_name[9:12])
        R = int(file_name[13:16])  # ignore
        A = int(file_name[17:20])
        return {'S': S, 'C': C, 'P': P, 'R': R, 'A': A}
    elif dataset == 'etri':
        A = int(file_name[1:4])
        P = int(file_name[5:8])
        G = int(file_name[9:12])  # ignore
        C = int(file_name[13:16])
        return {'A': A, 'P': P, 'G': G, 'C': C}


def build_group_key(parts, dataset):
    """
    Builds the grouping key. 
    For NTU/NTU120: key = (S, C).
    For ETRI: key = C.
    """
    if dataset in ['ntu', 'ntu120']:
        return (parts['S'], parts['C'])
    elif dataset == 'etri':
        return parts['C']


def organize_data(data, setting, dataset='ntu120', T=64):
    """
    Organizes data into train_data and test_data, grouped by either (S, C) or C,
    depending on dataset. Then precomputes actor->actions and file maps for sampling.
    
    Returns:
        train_data, test_data
        where train_data[group_key] and test_data[group_key] each contain:
            {
              'actor_to_actions': {actor: set_of_actions, ...},
              'pa_map': {(p, a): [list_of_fnames_for_that_p_a], ...},
              'actors_with_2plus': [actor1, actor2, ...]
            }
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
    def build_summary(pa_list):
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


def sample_data(organized_dict):
    """
    Randomly samples from the summarized dictionary (group_key -> summary).
    We look for 2 different actors that share at least 2 common actions.
    
    Returns a 4-tuple list of:
        (p1, a1, fname), (p1, a2, fname), (p2, a1, fname), (p2, a2, fname)
    or None if unsuccessful.
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
                    return [
                        (p1, a1, p1_a1_fname),
                        (p1, a2, p1_a2_fname),
                        (p2, a1, p2_a1_fname),
                        (p2, a2, p2_a2_fname)
                    ]
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
        else:
            failed_attempts += 1

    if failed_attempts >= max_failed_attempts:
        print('Failed to sample enough data without duplicates (single-thread).')
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
    if len(unique_results) < samples:
        print('Failed to generate enough unique samples (multi-thread).')
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
    def __init__(self, sampled_data, X):
        self.X = X  # The dictionary: fname -> skeleton array
        self.sampled_data = sampled_data  # List of 4-tuples
        # Extract actor pairs and action pairs for fast retrieval
        # Index 0 and 2 are the "same action" pairs in a sense, but 
        # typically we track p1, p2 or a1, a2. 
        # We'll store them in arrays for convenience:
        self.actors = np.array([[sample[0][0], sample[2][0]] for sample in sampled_data], dtype=float)
        self.actions = np.array([[sample[0][1], sample[2][1]] for sample in sampled_data], dtype=float)

    def __getitem__(self, index):
        sample = self.sampled_data[index]
        # 0: (p1, a1, fname)
        # 1: (p1, a2, fname)
        # 2: (p2, a1, fname)
        # 3: (p2, a2, fname)
        x1 = self.X[sample[0][2]]  # P1, A1
        x2 = self.X[sample[3][2]]  # P2, A2
        y1 = self.X[sample[1][2]]  # P1, A2
        y2 = self.X[sample[2][2]]  # P2, A1

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
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class Masked_AE_Data(Dataset):
    """
    Simple masked autoencoder style dataset.
    For each skeleton, randomly masks frames and joints.
    """
    def __init__(self, X, frame_masking_ratio=0.5, joint_masking_ratio=0.5):
        self.X = X
        self.frame_masking_ratio = frame_masking_ratio
        self.joint_masking_ratio = joint_masking_ratio

    def __getitem__(self, index):
        x = self.X[index].copy()  # Make a copy so we don't destroy original
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

        return x

    def __len__(self):
        return len(self.X)


def get_cross_data(X, dataset, setting, batch_size=32, return_loader=False, 
                   train_samples=50000, test_samples=5000, threads=1):
    """
    Entry point to generate 'paired' cross data (two actors, two actions) 
    for training and validation sets.

    Returns either (train_dataset, val_dataset) 
    or (train_loader, val_loader) if return_loader=True.
    """
    # 1) Organize data by group key
    organized_data_train, organized_data_test = organize_data(X, setting, dataset)

    # 2) Generate samples either single- or multi-threaded
    if threads == 1:
        train_data = gen_samples_single_threaded(train_samples, organized_data_train)
        val_data = gen_samples_single_threaded(test_samples, organized_data_test)
    else:
        train_data = gen_samples(train_samples, organized_data_train, threads=threads)
        val_data = gen_samples(test_samples, organized_data_test, threads=threads)

    # 3) Build the dataset objects
    train_dataset = Cross_Data(train_data, X)
    val_dataset = Cross_Data(val_data, X)

    # 4) Optionally wrap them in DataLoader
    if return_loader:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl

    return train_dataset, val_dataset
