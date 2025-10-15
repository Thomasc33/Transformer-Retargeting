import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

dmr_encoded_channels = (256, 32)
T=75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_dimension_conv = False
util_classifier_alpha = 10
priv_classifier_alpha = 0.1

# Input is size of latent space
class Adversary_Emb(nn.Module):
    def __init__(self, num_classes):
        super(Adversary_Emb, self).__init__()
        self.channels = [dmr_encoded_channels[0], 128, 256, 512]
        self.conv1 = nn.ConvTranspose1d(self.channels[0], self.channels[1], 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(self.channels[1], self.channels[2], 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(self.channels[2], self.channels[3], 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(self.channels[1])
        self.bn2 = nn.BatchNorm1d(self.channels[2])
        self.bn3 = nn.BatchNorm1d(self.channels[3])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.channels[3], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)
        # x = self.fc3(x)
        return x
    
class Discriminator(nn.Module): # 1 = real, 0 = fake
    def __init__(self):
        super(Discriminator, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        self.fc1 = nn.Linear(80, 32)
        self.fc2 = nn.Linear(32, 1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        #flatten
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


class DMR_Encoder1D(nn.Module):
    def __init__(self):
        super(DMR_Encoder1D, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=512, out_channels=dmr_encoded_channels[0], kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(dmr_encoded_channels[0], dmr_encoded_channels[0] * dmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1) 
        x = self.fc1(x)
        x = x.view(-1, *dmr_encoded_channels)

        return x

class DMR_Decoder1D(nn.Module):
    def __init__(self):
        super(DMR_Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=dmr_encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)

        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
 
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 

        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref1(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref2(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref3(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref4(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        return x
    
class DMR_Encoder2D(nn.Module):
    def __init__(self):
        super(DMR_Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=dmr_encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(dmr_encoded_channels[0], dmr_encoded_channels[0] * dmr_encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)

        x = self.ref(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, *dmr_encoded_channels)

        return x

class DMR_Decoder2D(nn.Module):
    def __init__(self):
        super(DMR_Decoder2D, self).__init__()

        self.dec1 = nn.ConvTranspose2d(in_channels=dmr_encoded_channels[0]*2, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(3,3), stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=96, out_channels=75, kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 
        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.ref(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)

        x = self.ref(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        
        return x
    
class DMR(nn.Module):
    def __init__(self, adv_lr=1e-4, use_adv=False, dataset='ntu60', batch_size=32, datasets={}):
        super(DMR, self).__init__()
        self.batch_size = batch_size

        # Dataset Info
        assert dataset in datasets.keys(), 'Dataset not found'
        self.utility_classes = datasets[dataset]['num_class']
        self.privacy_classes = datasets[dataset]['num_actor']

        # AutoEncoder Models
        if one_dimension_conv:
            self.static_encoder = DMR_Encoder1D()
            self.advamic_encoder = DMR_Encoder1D()
            self.decoder = DMR_Decoder1D()
        else:
            self.static_encoder = DMR_Encoder2D()
            self.dynamic_encoder = DMR_Encoder2D()
            self.decoder = DMR_Decoder1D()


        # Loss Functions
        self.triplet_loss = nn.TripletMarginLoss()
        self.bce_loss = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Info for loss functions
        self.end_effectors = torch.tensor([19, 15, 23, 24, 21, 22, 3]).to(device) * 3
        self.chain_lengths = torch.tensor([5, 5, 8, 8, 8, 8, 5]).to(device)

        # Lambdas for discounted loss
        self.lambda_rec = 2
        self.lambda_cross = 0.1
        self.lambda_ee = 1
        self.lambda_smoothing = 3
        self.lambda_trip = 1
        self.lambda_latent = 10
        self.lambda_adv_util_coop = util_classifier_alpha
        self.lambda_adv_priv_coop = priv_classifier_alpha
        self.lambda_adv_util_adv = util_classifier_alpha
        self.lambda_adv_priv_adv = priv_classifier_alpha
        self.lambda_adv_disc = 1

        # Loss Toggles
        self.use_rec_loss = True
        self.use_cross_loss = True
        self.use_ee_loss = True 
        self.use_trip_loss_paired = True 
        self.use_trip_loss_unpaired = True
        self.use_smoothing_loss = True
        self.use_latent_consistency = True

    def get_loss_params(self):
        return {
            'lambda_rec': self.lambda_rec,
            'lambda_cross': self.lambda_cross,
            'lambda_ee': self.lambda_ee,
            'lambda_trip': self.lambda_trip,
            'lambda_latent': self.lambda_latent,
            'lambda_adv_util_coop': self.lambda_adv_util_coop,
            'lambda_adv_priv_coop': self.lambda_adv_priv_coop,
            'lambda_adv_util_adv': self.lambda_adv_util_adv,
            'lambda_adv_priv_adv': self.lambda_adv_priv_adv,
            'lambda_adv_disc': self.lambda_adv_disc,
            'use_rec_loss': self.use_rec_loss,
            'use_cross_loss': self.use_cross_loss,
            'use_ee_loss': self.use_ee_loss,
            'use_trip_loss_paired': self.use_trip_loss_paired,
            'use_trip_loss_unpaired': self.use_trip_loss_unpaired,
            'use_smoothing_loss': self.use_smoothing_loss,
            'use_latent_consistency': self.use_latent_consistency
        }

    def cross(self, x1, x1_rot, x2, x2_rot):
        d1 = self.dynamic_encoder(x1_rot)
        d2 = self.dynamic_encoder(x2_rot)
        s1 = self.static_encoder(x1_rot)
        s2 = self.static_encoder(x2_rot)
        
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1))
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1))
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1))
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1))

        return x1_hat, x2_hat, y1_hat, y2_hat
    
    def eval(self, x1_rot, x2):
        dynamic = self.dynamic_encoder(x1_rot)
        static = self.static_encoder(x2)
        return self.decoder(torch.cat((dynamic, static), dim=1))

    def rec_loss(self, x, x_rot):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_rot)
        x_hat = self.decoder(torch.cat((d, s), dim=1))
        if not one_dimension_conv:
            x_ = x.reshape(x.size(0), T, -1)
        return self.reconstruction_loss(x_, x_hat)
    
    def loss_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, cross = True, reconstruction = True, emb_adv = True, discrim_adv = True, verbose = False):
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        s1 = self.static_encoder(x1_rot) # P1 - Use preprocessed data
        s2 = self.static_encoder(x2_rot) # P2 - Use preprocessed data

        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1)) # P2, A1
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1)) # P1, A2

        d12 = self.dynamic_encoder(y1_rot) # A1
        d21 = self.dynamic_encoder(y2_rot) # A2
        s12 = self.static_encoder(y1_rot) # P2 - Use preprocessed data
        s21 = self.static_encoder(y2_rot) # P1 - Use preprocessed data

        x1_hat_ = self.decoder(torch.cat((d12, s21), dim=1)) # P1, A1
        x2_hat_ = self.decoder(torch.cat((d21, s12), dim=1)) # P2, A2
        y1_hat_ = self.decoder(torch.cat((d12, s12), dim=1)) # P2, A1
        y2_hat_ = self.decoder(torch.cat((d21, s21), dim=1)) # P1, A2

        # x1_hat is reconstruction of x1
        # x2_hat is reconstruction of x2
        # y1_hat is cross reconstruction from x1 and x2
        # y2_hat is cross reconstruction from x2 and x1
        # x1_hat_ is cross reconstruction from y1 and y2
        # x2_hat_ is cross reconstruction from y2 and y1
        # y1_hat_ is reconstruction of y1
        # y2_hat_ is reconstruction of y2
        # d1 = A1
        # d2 = A2
        # d12 = A1
        # d21 = A2
        # s1 = P1
        # s2 = P2
        # s12 = P2
        # s21 = P1

        # flatten data if 2D
        if not one_dimension_conv:
            x1 = x1.view(x1.size(0), T, -1)
            x2 = x2.view(x2.size(0), T, -1)
            y1 = y1.view(y1.size(0), T, -1)
            y2 = y2.view(y2.size(0), T, -1)
        
        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        cross_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        latent_consistency_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)
                        
        # reconstruction loss
        if self.use_rec_loss and reconstruction:
            rec_loss = (self.reconstruction_loss(x1, x1_hat) + self.reconstruction_loss(x2, x2_hat) + self.reconstruction_loss(y1, y1_hat_) + self.reconstruction_loss(y2, y2_hat_)) / 4
            if verbose: print('Reconstruction Loss: ', rec_loss.item())
        
        # cross reconstruction loss
        if self.use_cross_loss and cross:
            # could move this to its own function, but since cross is basically reconstruction, its fine like this
            cross_loss = (self.reconstruction_loss(y1, y1_hat) + self.reconstruction_loss(y2, y2_hat) + self.reconstruction_loss(x1, x1_hat_) + self.reconstruction_loss(x2, x2_hat_)) / 4
            if verbose: print('Cross Reconstruction Loss: ', cross_loss.item())
        
        # end effector loss
        if self.use_ee_loss:
            if reconstruction:
                end_effector_loss += (self.end_effector_loss(x1_hat, x1) + self.end_effector_loss(x2_hat, x2)) / 2
            if cross:
                end_effector_loss += (self.end_effector_loss(y1_hat, y1) + self.end_effector_loss(y2_hat, y2)) / 2
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # triplet loss
        if self.use_trip_loss_paired: # anchor, positive, negative
            # d1 = A1, d2 = A2, d12 = A1, d21 = A2
            # s1 = P1, s2 = P2, s12 = P2, s21 = P1
            # d12,s12 = y1, d21,s21 = y2
            # y1 = jk, y2 = il
            triplet_loss = self.triplet_loss(d12, d1, d2) \
                            + self.triplet_loss(d21, d2, d1) \
                            + self.triplet_loss(s12, s2, s1) \
                            + self.triplet_loss(s21, s1, s2) 
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        if self.use_smoothing_loss:
            smoothing_loss = (self.smoothing_loss(x1, x1_hat) + self.smoothing_loss(x2, x2_hat) + self.smoothing_loss(y1, y1_hat_) + self.smoothing_loss(y2, y2_hat_) + \
                                self.smoothing_loss(x1, x1_hat_) + self.smoothing_loss(x2, x2_hat_) + self.smoothing_loss(y1, y1_hat) + self.smoothing_loss(y2, y2_hat)) / 8
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # latent consistency loss
        if self.use_latent_consistency:
            latent_consistency_loss = (self.latent_consistency_loss(d1, d12) + self.latent_consistency_loss(d2, d21) + self.latent_consistency_loss(s1, s21) + self.latent_consistency_loss(s2, s12)) / 4
            if verbose: print('Latent Consistency Loss: ', latent_consistency_loss.item())

        
        losses = {
            'rec_loss': rec_loss.item(),
            'cross_loss': cross_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'latent_consistency_loss': latent_consistency_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + cross_loss * self.lambda_cross \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + latent_consistency_loss * self.lambda_latent \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x1_hat, x2_hat, y1_hat, y2_hat, losses

    def loss_unpaired(self, x_pos, x_rot, actors, actions, reconstruction = True, emb_adv = False, discrim_adv = False, ee = False, triplet = False, verbose = False):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_rot) # Use preprocessed data
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        if not one_dimension_conv:
            x = x_pos.reshape(x_pos.size(0), T, -1)

        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        # Reconstruction Loss
        if self.use_rec_loss and reconstruction:
            rec_loss = self.reconstruction_loss(x, x_hat)
            if verbose: print('Reconstruction Loss: ', rec_loss.item())

        # End Effector Loss
        if self.use_ee_loss and ee:
            end_effector_loss = self.end_effector_loss(x_hat, x)
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # Triplet Loss
        if self.use_trip_loss_unpaired and triplet: # anchor, positive, negative
            triplet_loss = (self.triplet_loss(d, d, s) + self.triplet_loss(s, s, d)) / 2
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        # Smoothing Loss
        if self.use_smoothing_loss:
            smoothing_loss = self.smoothing_loss(x, x_hat)
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())


        losses = {
            'rec_loss': rec_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x_hat, losses

    def reconstruction_loss(self, x, y):
        # return F.mse_loss(x, y)
        return torch.square(torch.norm(x - y, dim=1)).mean()
    
    def latent_consistency_loss(self, x, y):
        return F.mse_loss(x, y)
    
    def end_effector_loss(self, x, y):
        # slice to get the end effector joints
        x_ee = x[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)] 
        y_ee = y[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)]

        # calculate velocities
        x_vel = torch.norm(x_ee[:, 1:] - x_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        y_vel = torch.norm(y_ee[:, 1:] - y_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        
        # compute mse loss for each joint
        losses = F.mse_loss(x_vel, y_vel, reduction='none')

        # take sum over end effectors
        loss = losses.sum(dim=1)

        # take mean over batch
        loss = loss.mean()
        
        return loss
    
    def smoothing_loss(self, y, y_pred):
        # (batch, T, 75)
        # Calculate the squared sum of differences for y and y_pred
        diff_y = torch.sum(y[:, :-1] - y[:, 1:], dim=2) ** 2
        diff_y_pred = torch.sum(y_pred[:, :-1] - y_pred[:, 1:], dim=2) ** 2

        # Calculate the absolute difference
        abs_diff = torch.abs(diff_y - diff_y_pred)

        # Sum over all batches and sequence elements
        loss = torch.sum(abs_diff)

        # Normalize by the total number of elements (batch_size * sequence_length)
        total_loss = torch.sqrt(loss) / (y.size(0) * y.size(1))

        return total_loss

    def adv_loss(self, model, x, y):
        return self.cross_entropy(model(x), y)#.long().to(device))
    
    def adv_accuracy(self, model, x, y):
        return (model(x).argmax(dim=1) == y.argmax(dim=1).to(device)).float().mean()

    def forward(self, x, x_rot):
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x_rot)
        x = self.decoder(torch.cat((dyn, sta), dim=1))
        return x
    
    def set_eval(self, eval=True):
        if eval:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()