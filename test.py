import torch

paired_data = torch.load('data/ntu_cs_paired.pt')
paired_train = paired_data['train']
paired_test = paired_data['test']

print(len(paired_train), len(paired_test))
print(type(paired_train), type(paired_test))

paired_train.sampled_data = paired_train.sampled_data[:300]
paired_test.sampled_data = paired_test.sampled_data[:300]

print(len(paired_train), len(paired_test))