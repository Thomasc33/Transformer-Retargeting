from data import load_data, get_cross_data, sample_data
import torch

# argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ntu')
parser.add_argument('--cpus-per-task', type=int, default=1)

args = parser.parse_args()

dataset = args.dataset
T=64
setting='cs'
batch_size=32
train_samples=10000
test_samples=5000
save_samples=True

paired_file_path = f'data/{dataset}_{setting}_paired.pt'
# Load data
X = load_data(dataset, T)
# Generate paired data and save to torch file
paired_train, paired_test = get_cross_data(X, dataset, setting, batch_size, return_loader=False, train_samples=train_samples, test_samples=test_samples, threads=args.cpus_per_task)
if save_samples:
    torch.save({'train': paired_train, 'test': paired_test}, paired_file_path)
    print('Paired data saved to torch file')