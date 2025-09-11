# Modified data loader for SGN that works with PKL files
import torch
import numpy as np
import pickle
import csv
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data import parse_file_name

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype=np.int32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

class Dataloaders(object):
    def __init__(self, dataset='NTU', case=0, seg=30, tag='ar'):

        # Set dataset name to lowercase for consistency with data.py
        self.dataset_lower = dataset.lower()

        # Load the appropriate data using the load_data function from data.py
        from data import load_data, datasets

        # Map dataset names to expected format in data.py
        dataset_mapping = {
            'NTU': 'ntu',
            'NTU120': 'ntu120',
            'ETRI': 'etri'
        }

        data_key = dataset_mapping.get(dataset, dataset.lower())

        # Determine number of classes
        if tag == 'gc':  # Gender classification is always binary
            num_classes = 2
            # Load gender data
            self.gender_map = self._load_gender_data(data_key)
            print(f"Loaded gender data for {len(self.gender_map)} subjects")
        elif data_key == 'ntu':
            num_classes = 60 if tag == 'ar' else 40  # 60 actions, 40 actors
        elif data_key == 'ntu120':
            num_classes = 120 if tag == 'ar' else 106  # 120 actions, 106 actors
        elif data_key == 'etri':
            num_classes = 55 if tag == 'ar' else 100  # 55 actions, 100 actors
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Loading data for {dataset} ({data_key})")
        if tag == 'ar':
            print("Task: Action Recognition")
        elif tag == 'ri':
            print("Task: Re-identification")
        elif tag == 'gc':
            print("Task: Gender Classification")
        print(f"Expected number of classes: {num_classes}")

        # Load the data using the existing load_data function
        data_dict = load_data(data_key, T=75)

        # Track unique labels for validation
        unique_actions = set()
        unique_subjects = set()

        self.dataset = dataset
        self.case = case
        self.tag = tag
        self.seg = seg
        train_x, train_y = [], []
        test_x, test_y = [], []

        # For re-identification, we cannot use cross-subject
        if tag == 'ri' and case == 0:
            raise ValueError("Re-identification cannot use cross-subject setting as actors must be seen during training")

        # For gender classification, we can only use NTU and NTU120
        if tag == 'gc' and dataset not in ['NTU', 'NTU120']:
            raise ValueError("Gender classification is only supported for NTU and NTU120 datasets")

        # Cross-setup is only available for NTU120
        if case == 2 and dataset != 'NTU120':
            raise ValueError(f"Cross-setup protocol only available for NTU120 dataset")

        # Define splits based on protocol
        if case == 0:  # Cross-subject
            if dataset == 'NTU' or dataset == 'NTU120':
                train_subjects = datasets[data_key]['train_actors']
                test_subjects = [i for i in range(1, 41) if i not in train_subjects]
            elif dataset == 'ETRI':
                train_subjects = datasets[data_key]['train_actors']
                test_subjects = [i for i in range(1, 101) if i not in train_subjects]

        elif case == 1:  # Cross-view
            if dataset == 'NTU' or dataset == 'NTU120':
                train_cameras = datasets[data_key]['train_cameras']
                test_cameras = datasets[data_key]['test_cameras']
            elif dataset == 'ETRI':
                train_cameras = datasets[data_key]['train_cameras']
                test_cameras = datasets[data_key]['test_cameras']

        elif case == 2:  # Cross-setup (NTU120 only)
            if dataset == 'NTU120':
                train_setups = list(range(1, 17))  # Setups 1-16
                test_setups = list(range(17, 33))  # Setups 17-32
            else:
                raise ValueError(f"Cross-setup protocol only available for NTU120 dataset")

        # Process sequence: translate, normalize and sample segments
        for file_name, skeleton_data in data_dict.items():
            parts = parse_file_name(file_name, dataset=self.dataset_lower)

            # We don't need to preprocess here as load_data already does the padding/truncating
            # Just reshape for the SGN format if needed

            # Always track subject IDs regardless of protocol
            subject_id = parts['P']
            unique_subjects.add(subject_id)

            # Split based on protocol
            if case == 0:  # Cross-subject
                if tag == 'ar':  # Action recognition: label = action
                    label = parts['A'] - 1  # 0-indexed
                    unique_actions.add(parts['A'])
                elif tag == 'ri':  # Re-identification: label = actor
                    label = parts['P'] - 1  # 0-indexed
                    if label >= num_classes:
                        continue  # Skip subjects beyond our class count
                elif tag == 'gc':  # Gender classification: label = gender (0 for female, 1 for male)
                    actor_id = parts['P']
                    if actor_id not in self.gender_map:
                        continue  # Skip subjects without gender information
                    # Convert gender to binary label (0 for female, 1 for male)
                    label = 1 if self.gender_map[actor_id] == 'M' else 0

                if subject_id in train_subjects:
                    train_x.append(skeleton_data)
                    train_y.append(label)
                elif subject_id in test_subjects:
                    test_x.append(skeleton_data)
                    test_y.append(label)

            elif case == 1:  # Cross-view
                camera_id = parts['C']
                if tag == 'ar':
                    label = parts['A'] - 1
                    unique_actions.add(parts['A'])
                elif tag == 'ri':
                    label = parts['P'] - 1
                    if label >= num_classes:
                        continue
                elif tag == 'gc':
                    actor_id = parts['P']
                    if actor_id not in self.gender_map:
                        continue
                    label = 1 if self.gender_map[actor_id] == 'M' else 0

                if camera_id in train_cameras:
                    train_x.append(skeleton_data)
                    train_y.append(label)
                elif camera_id in test_cameras:
                    test_x.append(skeleton_data)
                    test_y.append(label)

            elif case == 2:  # Cross-setup
                setup_id = parts['S']
                if tag == 'ar':
                    label = parts['A'] - 1
                    unique_actions.add(parts['A'])
                elif tag == 'ri':
                    label = parts['P'] - 1
                    if label >= num_classes:
                        continue
                elif tag == 'gc':
                    actor_id = parts['P']
                    if actor_id not in self.gender_map:
                        continue
                    label = 1 if self.gender_map[actor_id] == 'M' else 0

                if setup_id in train_setups:
                    train_x.append(skeleton_data)
                    train_y.append(label)
                elif setup_id in test_setups:
                    test_x.append(skeleton_data)
                    test_y.append(label)

        # Convert to numpy arrays
        self.train_x = np.array(train_x, dtype=np.float32)
        self.train_y = np.array(train_y, dtype=np.int64)
        self.test_x = np.array(test_x, dtype=np.float32)
        self.test_y = np.array(test_y, dtype=np.int64)

        # Create dataset objects
        self.train_set = Dataset(self.train_x, self.train_y)
        self.val_set = Dataset(self.test_x, self.test_y)
        self.test_set = Dataset(self.test_x, self.test_y)

        # Store sizes
        self.train_Y = self.train_y
        self.val_Y = self.test_y
        self.test_Y = self.test_y

        # Print validation information
        print("\nValidation Information:")
        if tag == 'ar':
            print(f"Number of unique actions found: {len(unique_actions)}")
            print(f"Action ID range: {min(unique_actions)} to {max(unique_actions)}")
        elif tag == 'ri':
            print(f"Number of unique subjects found: {len(unique_subjects)}")
            print(f"Subject ID range: {min(unique_subjects)} to {max(unique_subjects)}")
        elif tag == 'gc':
            # Count gender distribution in training set
            male_count = np.sum(self.train_y == 1)
            female_count = np.sum(self.train_y == 0)
            print(f"Gender distribution in training set: {female_count} female, {male_count} male")
            # Count gender distribution in test set
            male_count_test = np.sum(self.test_y == 1)
            female_count_test = np.sum(self.test_y == 0)
            print(f"Gender distribution in test set: {female_count_test} female, {male_count_test} male")
        print(f"Label range in train set: {min(self.train_y)} to {max(self.train_y)}\n")

        # Print data shapes
        print("Data Shapes:")
        print(f"Train data: {self.train_x.shape}, {self.train_y.shape}")
        print(f"Validation data: {self.test_x.shape}, {self.test_y.shape}")
        print(f"Test data: {self.test_x.shape}, {self.test_y.shape}\n")

        # Print first training sample info
        print("First training sample:")
        print(f"Input shape: {self.train_x[0].shape}")
        print(f"Label value: {self.train_y[0]}")

    def preprocess_sequence(self, skeleton_data):
        """
        Preprocess the skeleton data for SGN training:
        1. Use only first actor (75 dimensions = 25 joints * 3 channels)
        2. Remove zero frames (frames where all joint positions are zero)
        3. Sample segments according to SGN method

        Returns a sequence of length seg with properly sampled frames.
        """
        # Only use first actor for both tasks (25 joints * 3 channels = 75 dimensions)
        skeleton_data = skeleton_data[:, :75]

        # Remove zero frames (frames where all values are zero)
        non_zero_mask = ~np.all(skeleton_data == 0, axis=1)
        non_zero_frames = skeleton_data[non_zero_mask]

        # If sequence is shorter than seg frames, pad with zeros
        if len(non_zero_frames) < self.seg:
            pad = np.zeros((self.seg - len(non_zero_frames), non_zero_frames.shape[1]), dtype=np.float32)
            processed_seq = np.concatenate([non_zero_frames, pad], axis=0)
        else:
            # Sample frames with equal spacing as in the original SGN implementation
            num_frames = len(non_zero_frames)
            ave_duration = num_frames // self.seg

            # Sample frames at regular intervals with small random offset
            # This matches the original SGN implementation
            offsets = np.multiply(list(range(self.seg)), ave_duration) + np.random.randint(ave_duration, size=self.seg)

            # Ensure we don't go out of bounds
            offsets = np.clip(offsets, 0, num_frames-1)
            processed_seq = non_zero_frames[offsets]

        return processed_seq

    def get_train_loader(self, batch_size, num_workers):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=self.collate_fn_fix_train,
                          drop_last=True)

    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=self.collate_fn_fix_val,
                          drop_last=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=self.collate_fn_fix_test,
                          drop_last=True)

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def _load_gender_data(self, dataset):
        """Load gender data from CSV file"""
        gender_map = {}
        gender_file = f"data/{dataset}/statistics/Genders.csv"

        try:
            with open(gender_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert 1-indexed actor ID to int
                    actor_id = int(row['P'])
                    gender = row['Gender']
                    gender_map[actor_id] = gender
        except Exception as e:
            print(f"Error loading gender data: {e}")

        return gender_map

    def collate_fn_fix_train(self, batch):
        """
        Collate function for training data:
        1. Process each sequence to create segments of length seg
        2. Apply rotation augmentation
        """
        x, y = zip(*batch)
        processed_x = []

        # Process each sequence to create properly segmented data
        for seq in x:
            processed_seq = self.preprocess_sequence(seq)
            processed_x.append(processed_seq)

        # Stack the processed sequences
        x = torch.stack([torch.from_numpy(x_i) for x_i in processed_x], 0)
        theta = 0.4
        # Apply random rotation augmentation
        if self.case == 0:  # Cross-subject
            theta = 0.3
        elif self.case == 1:  # Cross-view
            theta = 0.5

        
        x = _transform(x, theta)
        y = torch.LongTensor(y)
        return [x, y]

    def collate_fn_fix_val(self, batch):
        """
        Collate function for validation data:
        1. Process each sequence to create segments of length seg
        2. No augmentation for validation
        """
        x, y = zip(*batch)
        processed_x = []

        # Process each sequence to create properly segmented data
        for seq in x:
            processed_seq = self.preprocess_sequence(seq)
            processed_x.append(processed_seq)

        # Stack the processed sequences
        x = torch.stack([torch.from_numpy(x_i) for x_i in processed_x], 0)
        y = torch.LongTensor(y)
        return [x, y]

    def collate_fn_fix_test(self, batch):
        """
        Collate function for test data:
        1. Creates 5 different augmented samples per sequence as in original SGN
        """
        x, y = zip(*batch)

        # Create 5 augmented samples for each sequence at test time
        x_augmented = []
        y_expanded = []

        for i, seq in enumerate(x):
            # Remove zero frames (frames where all values are zero)
            seq = seq[:, :75]  # Use only first actor (25 joints * 3 channels)
            non_zero_mask = ~np.all(seq == 0, axis=1)
            non_zero_frames = seq[non_zero_mask]

            # If sequence is too short, pad with zeros
            if len(non_zero_frames) < self.seg:
                pad = np.zeros((self.seg - len(non_zero_frames), non_zero_frames.shape[1]), dtype=np.float32)
                processed_seq = np.concatenate([non_zero_frames, pad], axis=0)
                # Just use one copy of this padded sequence 5 times
                for _ in range(5):
                    x_augmented.append(processed_seq)
                    y_expanded.append(y[i])
            else:
                # Following SGN's test-time augmentation with 5 different random samplings
                num_frames = len(non_zero_frames)
                ave_duration = num_frames // self.seg

                # Create 5 different random samplings
                for _ in range(5):
                    offsets = np.multiply(list(range(self.seg)), ave_duration) + np.random.randint(ave_duration, size=self.seg)
                    offsets = np.clip(offsets, 0, num_frames-1)
                    x_augmented.append(non_zero_frames[offsets])
                    y_expanded.append(y[i])

        x = torch.stack([torch.from_numpy(x_i) for x_i in x_augmented], 0)
        y = torch.LongTensor(y_expanded)
        return [x, y]

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test the data loader
    loader = Dataloaders('NTU', 0, seg=20, tag='ar')
    train_loader = loader.get_train_loader(32, 1)
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i}: {x.shape}, {y.shape}")
        if i == 2:
            break