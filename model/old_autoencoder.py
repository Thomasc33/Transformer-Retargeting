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
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))        

    def forward(self, x):
        N, C, T, V, M = x.size()
        if self.debug: print('enc 1 ', x.shape)
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        if self.debug: print('enc 2 ', x.shape)

        p = self.A_vector
        p = torch.tensor(p,dtype=torch.float)
        if self.debug: print('mat ', p.shape, x.shape)
        # print('p expanded ', p.to(x.device).expand(N*M*T, -1, -1).shape)
        if self.debug: print('n m ', N, M)
        x = p.to(x.device).expand(N*M*T, -1, -1) @ x
        if self.debug: print('enc 3 ', x.shape)
        
        x = self.to_joint_embedding(x)
        if self.debug: print('enc 4 ', x.shape)
        x += self.pos_embedding[:, :self.num_point]
        if self.debug: print('enc 5 ', x.shape)
        
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        if self.debug: print('enc 6 ', x.shape)
        x = self.data_bn(x)
        if self.debug: print('enc 7 ', x.shape)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        if self.debug: print('enc 8 ', x.shape)

        x = self.l1(x)
        if self.debug: print('enc 9 ', x.shape)
        x = self.l2(x)
        if self.debug: print('enc 10 ', x.shape)
        x = self.l3(x)
        if self.debug: print('enc 11 ', x.shape)
        x = self.l4(x)
        if self.debug: print('enc 12 ', x.shape)
        x2=x
        x = self.l5(x)
        if self.debug: print('enc 13 ', x.shape)
        x = self.l6(x)
        if self.debug: print('enc 14 ', x.shape)
        x = self.l7(x)
        if self.debug: print('enc 15 ', x.shape)
        x3=x
        x = self.l8(x)
        if self.debug: print('enc 16 ', x.shape)
        x = self.l9(x)
        if self.debug: print('enc 17 ', x.shape)
        x = self.l10(x)
        if self.debug: print('enc 18 ', x.shape)
                
        x2 = self.first_tram(x2)#x2(N*M,64,75,25)
        x3 = self.second_tram(x3)#x3(N*M,128,75,25)
        x =x + x2 + x3
        if self.debug: print('enc 19 ', x.shape)
        
        # x = x.reshape(N, M, 320, -1)
        # x = x.mean(3).mean(1)

        return x


class Decoder(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), out_channels=3, debug=False):
        super(Decoder, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, 8)
        self.num_point = num_point
        self.num_person = num_person
        self.debug = debug
        
        self.l10 = Ske_MixF(320, 320, A, 16)
        self.l9 = Ske_MixF(320, 320, A, 16)
        self.l8 = Ske_MixF(320, 160, A, 16)
        self.l8_upsample = nn.ConvTranspose2d(160, 160, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.l7 = Ske_MixF(160, 160, A, 32)
        self.l6 = Ske_MixF(160, 160, A, 32)
        self.l5 = Ske_MixF(160, 80, A, 32)
        self.l5_upsample = nn.ConvTranspose2d(80, 80, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))
        self.l4 = Ske_MixF(80, 80, A, 64)
        self.l3 = Ske_MixF(80, 80, A, 64)
        self.l2 = Ske_MixF(80, 80, A, 64)
        self.l1 = Ske_MixF(80, 80, A, 64, residual=False)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, 80))
        self.to_joint_embedding = nn.Linear(80, out_channels)
        
        self.data_bn = nn.BatchNorm1d(num_person * 80 * num_point)
        bn_init(self.data_bn, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
                
    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))  
    
    def forward(self, x):
        N = int(x.size(0)/2)
        M = self.num_person 
        if self.debug: print('dec 1 ', x.shape)

        x = self.l10(x)
        if self.debug: print('dec 2 ', x.shape)
        x = self.l9(x)
        if self.debug: print('dec 3 ', x.shape)
        x = self.l8(x)
        if self.debug: print('dec 4 ', x.shape)
        x = self.l8_upsample(x)
        if self.debug: print('dec 4_upsample ', x.shape)
        x = self.l7(x)
        if self.debug:  print('dec 5 ', x.shape)
        x = self.l6(x)
        if self.debug: print('dec 6 ', x.shape)
        x = self.l5(x)
        if self.debug: print('dec 7 ', x.shape)
        x = self.l5_upsample(x)
        if self.debug: print('dec 7_upsample ', x.shape)
        x = self.l4(x)
        if self.debug: print('dec 8 ', x.shape)
        x = self.l3(x)
        if self.debug: print('dec 9 ', x.shape)
        x = self.l2(x)
        if self.debug: print('dec 10 ', x.shape)
        x = self.l1(x)
        if self.debug: print('dec 11 ', x.shape)

        x = rearrange(x, '(n m) c t v -> n (m v c) t', m=M, v=self.num_point).contiguous()
        if self.debug: print('dec 12 ', x.shape)
        x = self.data_bn(x)
        if self.debug: print('dec 13 ', x.shape)
        x = rearrange(x, 'n (m v c) t -> (n m t) v c', m=M, v=self.num_point, t=64).contiguous()
        if self.debug: print('dec 14 ', x.shape)

        x += self.pos_embedding[:, :self.num_point]
        if self.debug: print('dec 15 ', x.shape)
        x = self.to_joint_embedding(x)
        if self.debug: print('dec 16 ', x.shape)

        p = self.A_vector
        p = torch.tensor(p, dtype=torch.float)
        if self.debug: print('mat ', p.shape, x.shape)
        # print('p expanded ', p.to(x.device).expand(N*M*64, -1, -1).shape)
        if self.debug: print('n m ', N, M)
        x = p.to(x.device).expand(N * M * 64, -1, -1) @ x
        if self.debug: print('dec 17 ', x.shape)

        x = rearrange(x, '(n m t) v c -> n c t v m', m=M, v=self.num_point, t=64, c=3).contiguous()
        if self.debug: print('dec 18 ', x.shape)

        return x


   

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, out_channels=3, debug=False):
        super(Model, self).__init__()
        self.encoder = Encoder(num_class=num_class, num_point=num_point, num_person=num_person, graph=graph, graph_args=graph_args, in_channels=in_channels, debug=debug)
        self.decoder = Decoder(num_class=num_class, num_point=num_point, num_person=num_person, graph=graph, graph_args=graph_args, out_channels=out_channels, debug=debug)
        self.debug = debug
        
    def forward(self, x):
        if self.debug: print('input x shape: ', x.shape)
        x = self.encoder(x)
        if self.debug: print('encoded x shape: ', x.shape)
        x = self.decoder(x)
        if self.debug: print('decoded x shape: ', x.shape)
        return x
    

if __name__ == '__main__':
    data = np.load('data\\ntu120\\NTU120_CSet.npz')

    train_x = torch.utils.data.DataLoader(dataset=data['x_train'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=True)
    test_x = torch.utils.data.DataLoader(dataset=data['x_test'],
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=True)

    model = Model(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    model = model.cuda()


    # Assuming 'train_x' is your DataLoader
    for x in train_x:
        print(x.shape)
        x = x.float().cuda()

        # Reduce to 64 frames
        x = x[:, :64]
        
        # Reshape x to (N, C, T, V, M)
        N = x.shape[0]
        T = x.shape[1]
        M = 2  # Number of people
        V = 25  # Number of joints
        C = 3  # Number of input channels
        
        # Ensure that M * V * C equals the last dimension of x
        assert M * V * C == x.shape[2], "Mismatch in the number of joints, channels, and people"
        
        # Reshape x
        x = x.view(N, T, M, V, C).permute(0, 4, 1, 3, 2)  # N, C, T, V, M
        print(x.shape)
        out = model(x)
        print(out.shape)
        break


    def format_data(x):
        x = x.float().cuda()
        x = x[:, :64]
        N = x.shape[0]
        T = x.shape[1]
        M = 2  # Number of people
        V = 25  # Number of joints
        C = 3  # Number of input channels
        assert M * V * C == x.shape[2], "Mismatch in the number of joints, channels, and people"
        x = x.view(N, T, M, V, C).permute(0, 4, 1, 3, 2)  # N, C, T, V, M
        return x

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the model to training mode
    model.train()

    # Define the number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Iterate over the training data
        for x in train_x:
            x = format_data(x)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(x)
            
            # Compute the loss
            loss = criterion(output, x)
            
            # Backward pass
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            # Update the running loss
            running_loss += loss.item()
        
        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}: Average Loss = {running_loss / len(train_x)}")