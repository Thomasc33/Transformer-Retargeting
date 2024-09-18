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
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, debug=False):
        super(Encoder, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, 8)
        self.num_point = num_point        
        self.data_bn = nn.BatchNorm1d(num_person * 80 * num_point)        
        self.to_joint_embedding = nn.Linear(in_channels, 80)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))
        self.debug = debug
        
        self.l1 = Ske_MixF(80, 80, A, 64, residual=False)
        self.l2 = Ske_MixF(80, 80, A, 64)
        self.l3 = Ske_MixF(80, 80, A, 64)
        self.l4 = Ske_MixF(80, 80, A, 64)
        self.l5 = Ske_MixF(80, 160, A, 32, stride=2)
        self.l6 = Ske_MixF(160, 160, A, 32)
        self.l7 = Ske_MixF(160, 160, A, 32)
        self.l8 = Ske_MixF(160, 320, A, 16, stride=2)
        self.l9 = Ske_MixF(320, 320, A, 16)
        self.l10= Ske_MixF(320, 320, A, 16)

        self.fc = nn.Linear(320, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        # Retrospect Model
        self.first_tram = nn.Sequential(
                nn.AvgPool2d((4,1)),
                nn.Conv2d(80, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
        self.second_tram = nn.Sequential(
                nn.AvgPool2d((2,1)),
                nn.Conv2d(160, 320, 1),
                nn.BatchNorm2d(320),
                nn.ReLU()
            )
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        self.num_class=num_class
        
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        A = I - np.linalg.matrix_power(A_outward, k)
        A = A.astype(np.float32)  # Ensure the NumPy array is of type float32
        return torch.from_numpy(A)  # This will be a FloatTensor
     

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()

        p = self.A_vector.to(x.device).expand(N*M*T, -1, -1)
        x = p @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]

        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        # Pass through encoder layers
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x2 = x
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x3 = x
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
                
        x2 = self.first_tram(x2)
        x3 = self.second_tram(x3)
        x = x + x2 + x3

        # Reshape x to (N, M, C', T', V')
        N_M, C_new, T_new, V_new = x.shape
        N = N_M // M
        self.T_new = T_new  # Store reduced temporal dimension
        self.V_new = V_new  # Store reduced spatial dimension
        
        # Continue as before
        x = x.view(N, M, C_new, T_new, V_new)
        x = x.mean(dim=1)  # Shape: (N, C_new, T_new, V_new)
        x = x.view(N, C_new, T_new * V_new)  # Flatten spatial and temporal dimensions
        x = x.permute(2, 0, 1).contiguous()  # Shape: (sequence_length, batch_size, d_model)

        return x
