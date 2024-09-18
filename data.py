import random
import pickle
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

datasets = {
    'ntu120': {
        'path': 'data\\ntu120\\ntu120.pkl',
        'max_actors': 1,
        'joints': 25,
        'channels': 3,
        'train_cameras': [2, 3],
        'test_cameras': [1],
        'train_actors': [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103],
    },
    'ntu': {
        'path': 'data\\ntu\\ntu.pkl',
        'max_actors': 1,
        'joints': 25,
        'channels': 3,
        'train_cameras': [2, 3],
        'test_cameras': [1],
        'train_actors': [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38],
    },
}

def load_data(dataset):
    assert dataset in datasets, f'Dataset {dataset} not found'
    with open(datasets[dataset]['path'], 'rb') as f:
        data = pickle.load(f)
    if datasets[dataset]['max_actors'] == 1:
        # Assuming shape (frames, joints * channels)
        data = {k: v[:, :datasets[dataset]['joints'] * datasets[dataset]['channels']] for k, v in data.items()}
    return data

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
    for file_name, content in data.items():
        content = content[:T]
        parts = parse_file_name(file_name)
        organized_data[parts['C']].append((parts['P'], parts['A'], content))
        if setting == 'cs':
            if parts['P'] in train_actors:
                train_data[parts['C']].append((parts['P'], parts['A'], content))
            else:
                test_data[parts['C']].append((parts['P'], parts['A'], content))

        
    if setting == 'cv':
        for camera in train_cameras:
            train_data[camera].extend(organized_data[camera])

        for camera in test_cameras:
            test_data[camera].extend(organized_data[camera])

    return train_data, test_data

def sample_data(organized_data):
    # Pick a random C pair
    C = random.choice(list(organized_data.keys()))

    # Get all (P, A, content) tuples for this C 
    pa_list = organized_data[C]

    # Pick 2 unique P values, find two overlapping A's
    

    # Pick 2 unique P values and 2 unique A values
    random.shuffle(pa_list)
    unique_p = set()
    unique_a = set()
    for p, a, _ in pa_list:
        if len(unique_p) < 2:
            unique_p.add(p)
        if len(unique_a) < 2:
            unique_a.add(a)
        if len(unique_p) == 2 and len(unique_a) == 2:
            break

    if len(unique_p) < 2 or len(unique_a) < 2:
        raise Exception(f'Not enough unique P or A values for C pair {C}')

    # Form all four (P, A) pairs and get the corresponding content
    sampled_data = [] #(p1, a1) (p1, a2) (p2, a1) (p2, a2)
    for p in unique_p:
        for a in unique_a:
            for pa_content in pa_list:
                if pa_content[0] == p and pa_content[1] == a:
                    sampled_data.append(pa_content)
                    break

    return sampled_data

def gen_samples(samples, data):
    d = []
    for _ in range(samples):
        failed = 0
        while True:
            d_ = sample_data(data)
            d_tuple = tuple(tuple(x) for x in d_)
            if len(d_tuple) == 4:
                d.append(d_)  # Add the unique sample to the dataset
                break
            failed += 1
            if failed > 100:
                print('failed to sample data')
                break
    return d

def sample_rec_data(X, dataset, setting, T=64):
    assert dataset in datasets, f'Dataset {dataset} not found'
    assert setting in ['cs', 'cv'], f'Setting {setting} not found'
    train_cameras = datasets[dataset]['train_cameras']
    train_actors = datasets[dataset]['train_actors']
    joints = datasets[dataset]['joints']
    channels = datasets[dataset]['channels']

    # Split by camera views
    X_train_keys = []
    X_test_keys = []
    if setting == 'cs':
        for key in X.keys():
            if parse_file_name(key)['P'] in train_actors:
                X_train_keys.append(key)
            else:
                X_test_keys.append(key)
    elif setting == 'cv':
        for key in X.keys():
            if parse_file_name(key)['C'] in train_cameras:
                X_train_keys.append(key)
            else:
                X_test_keys.append(key)
    
    # Create train and test sets
    X_train = np.zeros((len(X_train_keys), T, joints*channels))
    X_test = np.zeros((len(X_test_keys), T, joints*channels))
    for i, key in enumerate(X_train_keys):
        X_train[i] = X[key][:T]
    for i, key in enumerate(X_test_keys):
        X_test[i] = X[key][:T]

    # Get actor and action names
    train_actors = [parse_file_name(key)['P'] for key in X_train_keys]
    test_actors = [parse_file_name(key)['P'] for key in X_test_keys]
    train_actions = [parse_file_name(key)['A'] for key in X_train_keys]
    test_actions = [parse_file_name(key)['A'] for key in X_test_keys]
    
    return X_train, X_test, train_actors, train_actions, test_actors, test_actions

class Cross_Data(Dataset):
    def __init__(self, sampled_data):
        self.data = sampled_data  # The tuple is (actor, action, frames)
        # Extract and stack the content (frames) from the sampled data
        self.x1 = np.stack([sample[0][2] for sample in sampled_data])  # P1, A1
        self.x2 = np.stack([sample[3][2] for sample in sampled_data])  # P2, A2
        self.y1 = np.stack([sample[1][2] for sample in sampled_data])  # P1, A2
        self.y2 = np.stack([sample[2][2] for sample in sampled_data])  # P2, A1

        # Store actors and actions as NumPy arrays for easy retrieval
        self.actors = np.array(
            [[sample[0][0], sample[3][0]] for sample in sampled_data],
            dtype=float
        )
        self.actions = np.array(
            [[sample[0][1], sample[3][1]] for sample in sampled_data],
            dtype=float
        )

    def __getitem__(self, index):
        return (
            self.x1[index],
            self.x2[index],
            self.y1[index],
            self.y2[index],
            self.actors[index],
            self.actions[index]
        )

    def __len__(self):
        return len(self.x1)


class Rec_Data(Dataset):
    def __init__(self, X, Actor, Action):
        self.X = X
        self.Actor = Actor
        self.Action = Action
    
    def __getitem__(self, index):
        return self.X[index], float(self.Actor[index]), float(self.Action[index])
    
    def __len__(self):
        return len(self.X)
    

def get_cross_data(X, dataset, setting, batch_size=32, T=64, return_loader=True, train_samples=50000, test_samples=5000):
    organized_data_train, organized_data_test = organize_data(X, setting, dataset, T)
    train_data = gen_samples(train_samples, organized_data_train,)
    val_data = gen_samples(test_samples, organized_data_test)
    train_dataset = Cross_Data(train_data)
    val_dataset = Cross_Data(val_data)
    if return_loader:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl
    return train_dataset, val_dataset


def get_rec_data(X, dataset, setting, T=64, batch_size=32, return_loader=True):
    # Keep only 1000 samples of the data
    i = 0
    for key in list(X.keys()):
        if i < 1000:
            i += 1
        else:
            del X[key]
    X_train, X_test, train_actors, train_actions, test_actors, test_actions = sample_rec_data(X, dataset, setting, T)
    train_dataset = Rec_Data(X_train, train_actors, train_actions)
    val_dataset = Rec_Data(X_test, test_actors, test_actions)
    if return_loader:
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        return train_dl, val_dl
    return train_dataset, val_dataset