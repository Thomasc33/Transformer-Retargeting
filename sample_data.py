import logging
import sys
import torch
import argparse
from data import load_data, get_cross_data, sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'sample_data.log')
    ]
)
logger = logging.getLogger(__name__)

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ntu', help='Dataset to use (ntu, ntu120, etri)')
parser.add_argument('--cpus-per-task', type=int, default=1, help='Number of CPU threads to use for sampling')
parser.add_argument('--setting', type=str, default='cs', help='Evaluation setting (cs or cv)')
parser.add_argument('--train-samples', type=int, default=10000, help='Number of training samples to generate')
parser.add_argument('--test-samples', type=int, default=2000, help='Number of test samples to generate')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for data loading')
parser.add_argument('--no-save', action='store_true', help='Do not save the generated samples')

args = parser.parse_args()

dataset = args.dataset
T=64
setting=args.setting
batch_size=args.batch_size
train_samples=args.train_samples
test_samples=args.test_samples
save_samples=not args.no_save

logger.info(f"Starting data sampling for {dataset} dataset")
logger.info(f"Using {args.cpus_per_task} CPU threads for sampling")

paired_file_path = f'data/{dataset}_{setting}_paired_{train_samples}_{test_samples}.pt'

# Load data
logger.info(f"Loading {dataset} dataset...")
X = load_data(dataset, T)
logger.info(f"Loaded {len(X)} sequences from {dataset} dataset")

# Generate paired data and save to torch file
logger.info(f"Generating paired data with {train_samples} training samples and {test_samples} test samples")
paired_train, paired_test = get_cross_data(X, dataset, setting, batch_size, return_loader=False,
                                train_samples=train_samples, test_samples=test_samples,
                                threads=args.cpus_per_task, seg=T, augment=True,
                                train_theta=0.3 if setting == 'cs' else 0.5)

if save_samples:
    logger.info(f"Saving paired data to {paired_file_path}")
    torch.save({'train': paired_train, 'test': paired_test}, paired_file_path)
    logger.info('Paired data saved successfully')