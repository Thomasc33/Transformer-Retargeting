import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from einops import rearrange
from .tem_mixf import Temporal_MixFormer
from .spa_mixf import Spatial_MixFormer
from .ske_mixf import Ske_MixF, import_class, bn_init, conv_init

class Encoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3, debug=False, dataset='ntu'):
        super(Encoder, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, 8)
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.debug = debug
        self.dataset = dataset

        self.data_bn = nn.BatchNorm1d(self.num_person * 80 * self.num_point)
        self.to_joint_embedding = nn.Linear(self.in_channels, 80)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))

        # Encoder layers (Skeleton-MixFormer)
        self.l1 = Ske_MixF(80, 80, A, 64, residual=False)
        self.l2 = Ske_MixF(80, 80, A, 64)
        self.l3 = Ske_MixF(80, 80, A, 64)
        self.l4 = Ske_MixF(80, 80, A, 64)
        self.l5 = Ske_MixF(80, 160, A, 32, stride=2)
        self.l6 = Ske_MixF(160, 160, A, 32)
        self.l7 = Ske_MixF(160, 160, A, 32)
        self.l8 = Ske_MixF(160, 320, A, 16, stride=2)
        self.l9 = Ske_MixF(320, 320, A, 16)
        self.l10 = Ske_MixF(320, 320, A, 16)

        # Retrospect Model
        self.first_tram = nn.Sequential(
            nn.AvgPool2d((4, 1)),
            nn.Conv2d(80, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
        self.second_tram = nn.Sequential(
            nn.AvgPool2d((2, 1)),
            nn.Conv2d(160, 320, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )

        # Unfrozen embedding layer
        self.unfrozen_embedding = nn.Linear(320, 320)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.data_bn, 1)

        # Load pre-trained Skeleton-MixFormer weights
        pretrained_state_dict = torch.load(f'eval/mixformer/pretrained/{self.dataset}/ar.pth')
        model_state_dict = self.state_dict()

        # Remove 'module.' prefix if present in pretrained_state_dict keys
        pretrained_state_dict = {k.replace('module.', ''): v for k, v in pretrained_state_dict.items()}

        # Filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

        # Update model's state_dict
        model_state_dict.update(pretrained_state_dict)
        self.load_state_dict(model_state_dict)

        # Freeze the weights of the pre-trained layers
        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]:
            for param in layer.parameters():
                param.requires_grad = False

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        A = I - np.linalg.matrix_power(A_outward, k)
        A = A.astype(np.float32)
        return torch.from_numpy(A)

    def forward(self, x):
        N, C, T, V, M = x.size()  # Extract sizes

        # Pad zeros for a second actor
        if M == 1: 
            x = torch.cat([x, torch.zeros(N, C, T, V, 1).to(x.device)], dim=4)
            M = 2
        
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M).contiguous()

        p = self.A_vector.to(x.device).expand(N * M * T, -1, -1)
        x = torch.matmul(p, x)

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]

        # Specify 'n' and 't' explicitly
        x = rearrange(x, '(n m t) v c -> n (m v c) t', n=N, m=M, t=T).contiguous()
        x = self.data_bn(x)

        # Directly rearrange without adding extra dimension 's'
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', n=N, m=M, v=V).contiguous()

        # Pass through encoder layers (frozen)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2 = x  # Save for Retrospect Model
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3 = x  # Save for Retrospect Model
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # Apply Retrospect Model
        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x = x + x2 + x3

        # Reshape x to (N, M, C_new, T_new, V_new)
        N_M, C_new, T_new, V_new = x.shape
        N = N_M // M
        x = x.view(N, M, C_new, T_new, V_new)
        x = x.mean(dim=1)  # Average over persons if M > 1, shape: (N, C_new, T_new, V_new)

        # Apply unfrozen embedding layer
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T_new, V_new, C_new)
        x = x.view(N, T_new * V_new, C_new)     # (N, T_new * V_new, C_new)
        x = self.unfrozen_embedding(x)          # (N, T_new * V_new, 320)

        # Prepare for decoder
        x = x.permute(1, 0, 2).contiguous()     # (sequence_length, batch_size, d_model)

        return x


