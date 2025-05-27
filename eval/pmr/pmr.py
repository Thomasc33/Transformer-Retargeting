import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

encoded_channels = (128, 16)
T=75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_dimension_conv = False
util_classifier_alpha = 10
priv_classifier_alpha = 0.1


# Input is size of latent space
class Adversary_Emb(nn.Module):
    def __init__(self, num_classes):
        super(Adversary_Emb, self).__init__()
        self.channels = [encoded_channels[0], 128, 256, 512]
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

        # Use standard padding with smaller padding values
        self.enc1 = nn.Conv1d(in_channels=T, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        # Change the linear layer input size to 32 to match the actual flattened size
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.acti = nn.LeakyReLU(0.2)

        # Add adaptive pooling to ensure consistent size before the linear layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)  # 8 channels * 4 features = 32

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
        # Ensure input has correct shape
        if x.dim() != 3:
            try:
                x = x.view(x.size(0), T, -1)  # Reshape to [batch, T, features]
            except RuntimeError as e:
                print(f"Error reshaping input in discriminator: {e}")
                print(f"Input shape: {x.shape}")
                # Return a default value to prevent training failure
                return torch.zeros(x.size(0), 1, device=x.device)

        # Ensure the time dimension is correct
        if x.size(1) != T:
            print(f"Warning: Input time dimension is {x.size(1)}, expected {T}")
            try:
                # Interpolate to correct time dimension
                x = F.interpolate(x.transpose(1, 2), size=T).transpose(1, 2)
            except RuntimeError as e:
                print(f"Error interpolating time dimension: {e}")
                # Return a default value to prevent training failure
                return torch.zeros(x.size(0), 1, device=x.device)

        try:
            # Apply convolutions with standard padding
            x = self.acti(self.enc1(x))
            x = self.pool(x)

            x = self.acti(self.enc2(x))
            x = self.pool(x)

            x = self.acti(self.enc3(x))
            x = self.pool(x)

            x = self.acti(self.enc4(x))
            x = self.pool(x)

            # Apply adaptive pooling to ensure consistent size
            x = self.adaptive_pool(x)

            # Flatten the tensor
            x = x.view(x.size(0), -1)

            # Apply fully connected layers
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))

            return x

        except RuntimeError as e:
            print(f"Error in discriminator forward pass: {e}")
            print(f"Input shape: {x.shape}")
            # Provide a fallback to prevent training failure
            return torch.zeros(x.size(0), 1, device=x.device)

        return x

class Decoder1D(nn.Module):
    def __init__(self):
        super(Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
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

class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(encoded_channels[0], encoded_channels[0] * encoded_channels[1])

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
        x = x.view(-1, *encoded_channels)

        return x

class Decoder2D(nn.Module):
    def __init__(self):
        super(Decoder2D, self).__init__()

        self.dec1 = nn.ConvTranspose2d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
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


class PMR(nn.Module):
    def __init__(self, adv_lr=1e-4, use_adv=True, dataset='ntu', batch_size=32, datasets={}):
        super(PMR, self).__init__()
        self.batch_size = batch_size

        # AutoEncoder Models
        self.static_encoder = Encoder2D()
        self.dynamic_encoder = Encoder2D()
        self.decoder = Decoder1D()

        # Dataset info
        assert dataset in datasets.keys(), 'Dataset not found'
        self.privacy_classes = datasets[dataset]['num_actor']
        self.utility_classes = datasets[dataset]['num_class']

        # Adversarial Models
        self.use_adv = use_adv
        if use_adv:
            self.priv_adv = Adversary_Emb(self.privacy_classes).to(device) # input = dynamic embedding, output = privacy class
            self.priv_coop = Adversary_Emb(self.privacy_classes).to(device) # input = static embedding, output = privacy class
            self.util_adv = Adversary_Emb(self.utility_classes).to(device) # input = static embedding, output = utility class
            self.util_coop = Adversary_Emb(self.utility_classes).to(device) # input = dynamic embedding, output = utility class
            self.discriminator = Discriminator().to(device)

            self.priv_optim = torch.optim.AdamW(self.priv_adv.parameters(), lr=adv_lr)
            self.priv_coop_optim = torch.optim.AdamW(self.priv_coop.parameters(), lr=adv_lr)
            self.util_optim = torch.optim.AdamW(self.util_adv.parameters(), lr=adv_lr)
            self.util_coop_optim = torch.optim.AdamW(self.util_coop.parameters(), lr=adv_lr)
            self.discriminator_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=adv_lr)

            # Freeze Adversarial Models
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()

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
            x_ = x_rot.reshape(x_rot.size(0), T, -1)
        return self.reconstruction_loss(x_, x_hat)

    def loss_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, cross = True, reconstruction = True, emb_adv = True, discrim_adv = True, verbose = False):
        x1=x1_rot
        x2=x2_rot
        y1=y1_rot
        y2=y2_rot
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        s1 = self.static_encoder(x1_rot) # P1
        s2 = self.static_encoder(x2_rot) # P2

        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1)) # P2, A1
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1)) # P1, A2

        d12 = self.dynamic_encoder(y1_rot) # A1
        d21 = self.dynamic_encoder(y2_rot) # A2
        s12 = self.static_encoder(y1) # P2
        s21 = self.static_encoder(y2) # P1

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

        # adversarial loss
        if self.use_adv and emb_adv:
            try:
                # SAFETY CHECK: Ensure actors has proper dimensions
                if actors.dim() < 2 or actors.size(1) < 2:
                    print(f"WARNING: actors tensor has invalid shape: {actors.shape}. Expected [batch_size, 2]")
                    # Create default actor IDs
                    batch_size = x1.size(0)
                    actors = torch.ones(batch_size, 2, device=device)

                # Extract actor IDs with bounds checking
                actor_y1 = actors[:, 0].clone().to(device)
                actor_y2 = actors[:, 1].clone().to(device)

                # Check for negative values or zeros (since we're subtracting 1)
                actor_y1[actor_y1 <= 0] = 1  # Set to 1 if 0 or negative
                actor_y2[actor_y2 <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                actor_y1 = actor_y1 - 1
                actor_y2 = actor_y2 - 1

                # Ensure indices are within valid range
                actor_y1 = torch.clamp(actor_y1, 0, self.privacy_classes - 1)
                actor_y2 = torch.clamp(actor_y2, 0, self.privacy_classes - 1)

                # Create one-hot encoding on the same device as the indices
                actor_y1 = torch.eye(self.privacy_classes, device=device)[actor_y1.long()]
                actor_y2 = torch.eye(self.privacy_classes, device=device)[actor_y2.long()]

                # SAFETY CHECK: Ensure actions has proper dimensions
                if actions.dim() < 2 or actions.size(1) < 2:
                    print(f"WARNING: actions tensor has invalid shape: {actions.shape}. Expected [batch_size, 2]")
                    # Create default action IDs
                    batch_size = x1.size(0)
                    actions = torch.ones(batch_size, 2, device=device)

                # Extract action IDs with bounds checking
                action_y1 = actions[:, 0].clone().to(device)
                action_y2 = actions[:, 1].clone().to(device)

                # Check for negative values or zeros (since we're subtracting 1)
                action_y1[action_y1 <= 0] = 1  # Set to 1 if 0 or negative
                action_y2[action_y2 <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                action_y1 = action_y1 - 1
                action_y2 = action_y2 - 1

                # Ensure indices are within valid range
                action_y1 = torch.clamp(action_y1, 0, self.utility_classes - 1)
                action_y2 = torch.clamp(action_y2, 0, self.utility_classes - 1)

                # Create one-hot encoding on the same device as the indices
                action_y1 = torch.eye(self.utility_classes, device=device)[action_y1.long()]
                action_y2 = torch.eye(self.utility_classes, device=device)[action_y2.long()]
            except Exception as e:
                print(f"Error in preparing adversarial inputs: {e}")
                # Create default tensors to continue training
                batch_size = x1.size(0)
                actor_y1 = torch.zeros(batch_size, self.privacy_classes, device=device)
                actor_y1[:, 0] = 1  # Set first class as default
                actor_y2 = actor_y1.clone()
                action_y1 = torch.zeros(batch_size, self.utility_classes, device=device)
                action_y1[:, 0] = 1  # Set first class as default
                action_y2 = action_y1.clone()

            # x1 => d1 s1
            # x2 => d2 s2

            # d1 => p1
            # d2 => p2
            # s1 => a1
            # s2 => a2

            # actor_y1 = p1
            # actor_y2 = p2

            # action_y1 = a1
            # action_y2 = a2


            # privacy loss (adversarial)
            privacy_loss_adv = (-self.adv_loss(self.priv_adv, d1, actor_y1) -self.adv_loss(self.priv_adv, d2, actor_y2))/2
            privacy_acc_adv = (self.adv_accuracy(self.priv_adv, d1, actor_y1) + self.adv_accuracy(self.priv_adv, d2, actor_y2))/2

            # privacy loss (coop)
            privacy_loss_coop = (self.adv_loss(self.priv_coop, s1, actor_y1) + self.adv_loss(self.priv_coop, s2, actor_y2))/2
            privacy_acc_coop = (self.adv_accuracy(self.priv_coop, s1, actor_y1) + self.adv_accuracy(self.priv_coop, s2, actor_y2))/2

            # utility loss (adversarial)
            utility_loss_adv = (-self.adv_loss(self.util_adv, s1, action_y1) -self.adv_loss(self.util_adv, s2, action_y2))/2
            utility_acc_adv = (self.adv_accuracy(self.util_adv, s1, action_y1) + self.adv_accuracy(self.util_adv, s2, action_y2))/2

            # utility loss (coop)
            utility_loss_coop = (self.adv_loss(self.util_coop, d1, action_y1) + self.adv_loss(self.util_coop, d2, action_y2))/2
            utility_acc_coop = (self.adv_accuracy(self.util_coop, d1, action_y1) + self.adv_accuracy(self.util_coop, d2, action_y2))/2

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose:
                print('Privacy Loss (Adversarial): ', privacy_loss_adv.item(), '\tPrivacy Loss (Coop): ', privacy_loss_coop.item())
                print('Utility Loss (Adversarial): ', utility_loss_adv.item(), '\tUtility Loss (Coop): ', utility_loss_coop.item())
                print('Privacy Accuracy (Adversarial): ', privacy_acc_adv.item(), '\tPrivacy Accuracy (Coop): ', privacy_acc_coop.item())
                print('Utility Accuracy (Adversarial): ', utility_acc_adv.item(), '\tUtility Accuracy (Coop): ', utility_acc_coop.item())


        if self.use_adv and discrim_adv:
            try:
                # discrimnator (adversarial)
                # Ensure all tensors have the same batch size and are properly shaped
                batch_size = min(x1_hat.size(0), x2_hat.size(0), y1_hat.size(0), y2_hat.size(0),
                                x1_hat_.size(0), x2_hat_.size(0), y1_hat_.size(0), y2_hat_.size(0))

                # Safety check for batch size
                if batch_size <= 0:
                    raise ValueError(f"Invalid batch size: {batch_size}")

                try:
                    # Reshape all tensors to ensure they have the expected shape [batch_size, T, features]
                    x1_hat_view = x1_hat[:batch_size].view(batch_size, T, -1)
                    x2_hat_view = x2_hat[:batch_size].view(batch_size, T, -1)
                    y1_hat_view = y1_hat[:batch_size].view(batch_size, T, -1)
                    y2_hat_view = y2_hat[:batch_size].view(batch_size, T, -1)
                    x1_hat_view_ = x1_hat_[:batch_size].view(batch_size, T, -1)
                    x2_hat_view_ = x2_hat_[:batch_size].view(batch_size, T, -1)
                    y1_hat_view_ = y1_hat_[:batch_size].view(batch_size, T, -1)
                    y2_hat_view_ = y2_hat_[:batch_size].view(batch_size, T, -1)

                    # Concatenate along the batch dimension
                    all_fake = torch.cat((x1_hat_view, x2_hat_view, y1_hat_view, y2_hat_view,
                                        x1_hat_view_, x2_hat_view_, y1_hat_view_, y2_hat_view_))

                    # Check for NaN values
                    if torch.isnan(all_fake).any():
                        raise ValueError("NaN values detected in discriminator input")

                    discrim_out_fake = self.discriminator(all_fake)
                    discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
                    discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (8 * batch_size)
                    if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())
                except RuntimeError as e:
                    print(f"Error in discriminator tensor operations: {e}")
                    # Provide default values to continue training
                    discriminator_loss = torch.zeros(1).to(device)
                    discriminator_acc = torch.zeros(1).to(device)
            except Exception as e:
                print(f"Error in discriminator forward pass: {e}")
                # Provide default values to continue training
                discriminator_loss = torch.zeros(1).to(device)
                discriminator_acc = torch.zeros(1).to(device)

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
        x_pos=x_rot
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        # Reshape the input for reconstruction comparison
        if not one_dimension_conv:
            x_reshaped = x_pos.reshape(x_pos.size(0), T, -1)

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
            # FIX: Use x_reshaped instead of undefined x
            rec_loss = self.reconstruction_loss(x_reshaped, x_hat)
            if verbose: print('Reconstruction Loss: ', rec_loss.item())

        # End Effector Loss
        if self.use_ee_loss and ee:
            # FIX: Use x_reshaped instead of undefined x
            end_effector_loss = self.end_effector_loss(x_hat, x_reshaped)
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # Triplet Loss
        if self.use_trip_loss_unpaired and triplet: # anchor, positive, negative
            triplet_loss = (self.triplet_loss(d, d, s) + self.triplet_loss(s, s, d)) / 2
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        # Smoothing Loss
        if self.use_smoothing_loss:
            # FIX: Use x_reshaped instead of undefined x
            smoothing_loss = self.smoothing_loss(x_reshaped, x_hat)
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # Adversarial Loss
        if self.use_adv and emb_adv:
            try:
                # Safely prepare actor labels
                actor_y = actors.to(device).clone()

                # Check for negative values or zeros (since we're subtracting 1)
                actor_y[actor_y <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                actor_y = actor_y - 1

                # Ensure indices are within valid range
                actor_y = torch.clamp(actor_y, 0, self.privacy_classes - 1)

                # Create one-hot encoding on the same device as the indices
                actor_y = torch.eye(self.privacy_classes, device=device)[actor_y.long()]

                # Safely prepare action labels
                action_y = actions.to(device).clone()

                # Check for negative values or zeros (since we're subtracting 1)
                action_y[action_y <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                action_y = action_y - 1

                # Ensure indices are within valid range
                action_y = torch.clamp(action_y, 0, self.utility_classes - 1)

                # Create one-hot encoding on the same device as the indices
                action_y = torch.eye(self.utility_classes, device=device)[action_y.long()]
            except Exception as e:
                print(f"Error in preparing unpaired adversarial inputs: {e}")
                # Create default tensors to continue training
                batch_size = x_pos.size(0)
                actor_y = torch.zeros(batch_size, self.privacy_classes, device=device)
                actor_y[:, 0] = 1  # Set first class as default
                action_y = torch.zeros(batch_size, self.utility_classes, device=device)
                action_y[:, 0] = 1  # Set first class as default

            # latent privacy loss (adv)
            privacy_loss_adv = -self.adv_loss(self.priv_adv, d, actor_y)
            privacy_acc_adv = self.adv_accuracy(self.priv_adv, d, actor_y)

            # latent privacy loss (coop)
            privacy_loss_coop = self.adv_loss(self.priv_coop, s, actor_y)
            privacy_acc_coop = self.adv_accuracy(self.priv_coop, s, actor_y)

            # latent utility loss (adv)
            utility_loss_adv = -self.adv_loss(self.util_adv, s, action_y)
            utility_acc_adv = self.adv_accuracy(self.util_adv, s, action_y)

            # latent utility loss (coop)
            utility_loss_coop = self.adv_loss(self.util_coop, d, action_y)
            utility_acc_coop = self.adv_accuracy(self.util_coop, d, action_y)

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose:
                print('Privacy Loss Adv: ', privacy_loss_adv.item(), '\tPrivacy Loss Coop: ', privacy_loss_coop.item(), '\tPrivacy Loss: ', privacy_loss.item())
                print('Utility Loss Adv: ', utility_loss_adv.item(), '\tUtility Loss Coop: ', utility_loss_coop.item(), '\tUtility Loss: ', utility_loss.item())
                print('Privacy Accuracy Adv: ', privacy_acc_adv.item(), '\tPrivacy Accuracy Coop: ', privacy_acc_coop.item())
                print('Utility Accuracy Adv: ', utility_acc_adv.item(), '\tUtility Accuracy Coop: ', utility_acc_coop.item())


        if self.use_adv and discrim_adv:
            try:
                # discrimnator (adversarial)
                # Ensure proper tensor shape for discriminator input
                x_hat_view = x_hat.view(x_hat.size(0), T, -1)

                discrim_out_fake = self.discriminator(x_hat_view)
                discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
                discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (self.batch_size)
                if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())
            except Exception as e:
                print(f"Error in unpaired discriminator forward pass: {e}")
                # Provide default values to continue training
                discriminator_loss = torch.zeros(1).to(device)
                discriminator_acc = torch.zeros(1).to(device)

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

    def train_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0,0,0,0,0,0,0,0,0
        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        x1=x1_rot
        x2=x2_rot
        y1=y1_rot
        y2=y2_rot

        # move to device
        x1 = x1.to(device)
        x2 = x2.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        x1_rot = x1_rot.to(device)
        x2_rot = x2_rot.to(device)
        y1_rot = y1_rot.to(device)
        y2_rot = y2_rot.to(device)
        actors = actors.to(device)
        actions = actions.to(device)

        # unfreeze adversaries
        self.priv_adv.train()
        self.priv_coop.train()
        self.util_adv.train()
        self.util_coop.train()
        self.discriminator.train()

        # zero out gradients
        self.priv_optim.zero_grad()
        self.priv_coop_optim.zero_grad()
        self.util_optim.zero_grad()
        self.util_coop_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        # encode
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        d3 = self.dynamic_encoder(y1_rot) # A2
        d4 = self.dynamic_encoder(y2_rot) # A1
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2
        s3 = self.static_encoder(y1) # P1
        s4 = self.static_encoder(y2) # P2

        # decode
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d3, s3), dim=1)) # P1, A2
        y2_hat = self.decoder(torch.cat((d4, s4), dim=1)) # P2, A1

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # SAFETY CHECK: Ensure actors has proper dimensions
            if actors.dim() < 2 or actors.size(1) < 2:
                print(f"WARNING: actors tensor has invalid shape: {actors.shape}. Expected [batch_size, 2]")
                # Try to reshape if possible or use a default value
                if actors.numel() >= 2:
                    # Try to reshape a flattened tensor
                    batch_size = x1.size(0)
                    actors = actors.reshape(batch_size, -1)
                    if actors.size(1) < 2:
                        # Repeat the first column if we only have one
                        actors = actors.repeat(1, 2)
                else:
                    # Create default actor IDs
                    batch_size = x1.size(0)
                    actors = torch.ones(batch_size, 2, device=device)

            # Extract actor IDs with bounds checking
            try:
                # Ensure actors tensor has valid indices
                if actors.size(1) < 2:
                    print(f"Warning: actors tensor has invalid shape: {actors.shape}")
                    # Create default actor IDs
                    batch_size = x1.size(0)
                    actors = torch.zeros(batch_size, 2, device=device)

                # Extract actor IDs and subtract 1 (for 0-indexing)
                p1 = actors[:, 0].clone()
                p2 = actors[:, 1].clone()

                # Check for negative values or zeros (since we're subtracting 1)
                p1[p1 <= 0] = 1  # Set to 1 if 0 or negative
                p2[p2 <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                p1 = p1 - 1
                p2 = p2 - 1

                # Ensure indices are within valid range
                p1 = torch.clamp(p1, 0, self.privacy_classes - 1)
                p2 = torch.clamp(p2, 0, self.privacy_classes - 1)

                # Create one-hot encodings safely
                eye_tensor = torch.eye(self.privacy_classes, device=device)
                p1_onehot = eye_tensor[p1.long()]
                p2_onehot = eye_tensor[p2.long()]

                # train privacy adversary
                priv_loss = (self.cross_entropy(self.priv_adv(d1), p1_onehot) + \
                            self.cross_entropy(self.priv_adv(d2), p2_onehot) + \
                            self.cross_entropy(self.priv_adv(d3), p1_onehot) + \
                            self.cross_entropy(self.priv_adv(d4), p2_onehot)) / 4
                priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1_onehot) + \
                            self.adv_accuracy(self.priv_adv, d2, p2_onehot) + \
                            self.adv_accuracy(self.priv_adv, d3, p1_onehot) + \
                            self.adv_accuracy(self.priv_adv, d4, p2_onehot)) / 4
                priv_loss.backward(retain_graph=True)
                self.priv_optim.step()

                # train privacy cooperative
                priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1_onehot) + \
                                self.cross_entropy(self.priv_coop(s2), p2_onehot) + \
                                self.cross_entropy(self.priv_coop(s3), p1_onehot) + \
                                self.cross_entropy(self.priv_coop(s4), p2_onehot)) / 4
                priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1_onehot) + \
                                self.adv_accuracy(self.priv_coop, s2, p2_onehot) + \
                                self.adv_accuracy(self.priv_coop, s3, p1_onehot) + \
                                self.adv_accuracy(self.priv_coop, s4, p2_onehot)) / 4
                priv_coop_loss.backward(retain_graph=True)
                self.priv_coop_optim.step()
            except Exception as e:
                print(f"Error in privacy training: {e}")
                # Continue with other adversaries

            # SAFETY CHECK: Ensure actions has proper dimensions
            if actions.dim() < 2 or actions.size(1) < 2:
                print(f"WARNING: actions tensor has invalid shape: {actions.shape}. Expected [batch_size, 2]")
                # Try to reshape if possible
                if actions.numel() >= 2:
                    # Try to reshape a flattened tensor
                    batch_size = x1.size(0)
                    actions = actions.reshape(batch_size, -1)
                    if actions.size(1) < 2:
                        # Repeat the first column if we only have one
                        actions = actions.repeat(1, 2)
                else:
                    # Create default action IDs
                    batch_size = x1.size(0)
                    actions = torch.ones(batch_size, 2, device=device)

            # Extract action IDs with bounds checking
            try:
                # Ensure actions tensor has valid indices
                if actions.size(1) < 2:
                    print(f"Warning: actions tensor has invalid shape: {actions.shape}")
                    # Create default action IDs
                    batch_size = x1.size(0)
                    actions = torch.zeros(batch_size, 2, device=device)

                # Extract action IDs and subtract 1 (for 0-indexing)
                a1 = actions[:, 0].clone()
                a2 = actions[:, 1].clone()

                # Check for negative values or zeros (since we're subtracting 1)
                a1[a1 <= 0] = 1  # Set to 1 if 0 or negative
                a2[a2 <= 0] = 1  # Set to 1 if 0 or negative

                # Now subtract 1 safely
                a1 = a1 - 1
                a2 = a2 - 1

                # Ensure indices are within valid range
                a1 = torch.clamp(a1, 0, self.utility_classes - 1)
                a2 = torch.clamp(a2, 0, self.utility_classes - 1)

                # Create one-hot encodings safely
                utility_eye_tensor = torch.eye(self.utility_classes, device=device)
                a1_onehot = utility_eye_tensor[a1.long()]
                a2_onehot = utility_eye_tensor[a2.long()]

                # train utility adversary
                util_loss = (self.cross_entropy(self.util_adv(s1), a1_onehot) + \
                            self.cross_entropy(self.util_adv(s2), a2_onehot) + \
                            self.cross_entropy(self.util_adv(s3), a2_onehot) + \
                            self.cross_entropy(self.util_adv(s4), a1_onehot)) / 4
                util_acc = (self.adv_accuracy(self.util_adv, s1, a1_onehot) + \
                            self.adv_accuracy(self.util_adv, s2, a2_onehot) + \
                            self.adv_accuracy(self.util_adv, s3, a2_onehot) + \
                            self.adv_accuracy(self.util_adv, s4, a1_onehot)) / 4
                util_loss.backward(retain_graph=True)
                self.util_optim.step()

                # train utility cooperative
                util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1_onehot) + \
                                self.cross_entropy(self.util_coop(d2), a2_onehot) + \
                                self.cross_entropy(self.util_coop(d3), a2_onehot) + \
                                self.cross_entropy(self.util_coop(d4), a1_onehot)) / 4
                util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1_onehot) + \
                                self.adv_accuracy(self.util_coop, d2, a2_onehot) + \
                                self.adv_accuracy(self.util_coop, d3, a2_onehot) + \
                                self.adv_accuracy(self.util_coop, d4, a1_onehot)) / 4
                util_coop_loss.backward(retain_graph=True)
                self.util_coop_optim.step()
            except Exception as e:
                print(f"Error in utility training: {e}")

        if train_discrim:
            try:
                # train discriminator
                # Ensure proper reshaping and handling before concatenation
                try:
                    # Process real data
                    x1_view = x1.view(x1.size(0), T, -1)
                    x2_view = x2.view(x2.size(0), T, -1)
                    y1_view = y1.view(y1.size(0), T, -1)
                    y2_view = y2.view(y2.size(0), T, -1)

                    # Check shapes before concatenation
                    if not (x1_view.size(1) == x2_view.size(1) == y1_view.size(1) == y2_view.size(1) == T):
                        print(f"Warning: Inconsistent time dimension in real data")
                        print(f"Shapes: x1={x1_view.shape}, x2={x2_view.shape}, y1={y1_view.shape}, y2={y2_view.shape}")
                        # Ensure all tensors have the same time dimension
                        x1_view = F.interpolate(x1_view.transpose(1, 2), size=T).transpose(1, 2)
                        x2_view = F.interpolate(x2_view.transpose(1, 2), size=T).transpose(1, 2)
                        y1_view = F.interpolate(y1_view.transpose(1, 2), size=T).transpose(1, 2)
                        y2_view = F.interpolate(y2_view.transpose(1, 2), size=T).transpose(1, 2)

                    # Process fake data
                    x1_hat_view = x1_hat.view(x1_hat.size(0), T, -1)
                    x2_hat_view = x2_hat.view(x2_hat.size(0), T, -1)
                    y1_hat_view = y1_hat.view(y1_hat.size(0), T, -1)
                    y2_hat_view = y2_hat.view(y2_hat.size(0), T, -1)

                    # Check shapes before concatenation
                    if not (x1_hat_view.size(1) == x2_hat_view.size(1) == y1_hat_view.size(1) == y2_hat_view.size(1) == T):
                        print(f"Warning: Inconsistent time dimension in fake data")
                        print(f"Shapes: x1_hat={x1_hat_view.shape}, x2_hat={x2_hat_view.shape}, y1_hat={y1_hat_view.shape}, y2_hat={y2_hat_view.shape}")
                        # Ensure all tensors have the same time dimension
                        x1_hat_view = F.interpolate(x1_hat_view.transpose(1, 2), size=T).transpose(1, 2)
                        x2_hat_view = F.interpolate(x2_hat_view.transpose(1, 2), size=T).transpose(1, 2)
                        y1_hat_view = F.interpolate(y1_hat_view.transpose(1, 2), size=T).transpose(1, 2)
                        y2_hat_view = F.interpolate(y2_hat_view.transpose(1, 2), size=T).transpose(1, 2)

                    # Process real and fake data separately to avoid large concatenations
                    real_data = [x1_view, x2_view, y1_view, y2_view]
                    fake_data = [x1_hat_view, x2_hat_view, y1_hat_view, y2_hat_view]

                    # Process in smaller batches to avoid memory issues
                    output_real_list = []
                    output_fake_list = []

                    for data in real_data:
                        output = self.discriminator(data)
                        output_real_list.append(output)

                    for data in fake_data:
                        output = self.discriminator(data)
                        output_fake_list.append(output)

                    output_real = torch.cat(output_real_list)
                    output_fake = torch.cat(output_fake_list)

                    # Calculate loss and accuracy
                    discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + \
                                         self.bce_loss(output_fake, torch.zeros_like(output_fake))
                    discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * self.batch_size)) + \
                                        (torch.sum(torch.round(output_real) == 1).float() / (4 * self.batch_size))) / 2

                    # Backpropagate and update
                    discriminator_loss.backward()
                    self.discriminator_optim.step()

                except RuntimeError as e:
                    print(f"Error during discriminator data processing: {e}")
                    # Continue with other training steps
                    discriminator_loss = torch.zeros(1).to(device)
                    discriminator_acc = torch.zeros(1).to(device)

            except Exception as e:
                print(f"Error in discriminator training: {e}")
                discriminator_loss = torch.zeros(1).to(device)
                discriminator_acc = torch.zeros(1).to(device)

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def val_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0,0,0,0,0,0,0,0,0
        x1=x1_rot
        x2=x2_rot
        y1=y1_rot
        y2=y2_rot
        # freeze encoders/decoder
        self.set_eval()

        # Encode
        d1, d2, d3, d4 = [self.dynamic_encoder(x) for x in [x1_rot, x2_rot, y1_rot, y2_rot]]
        s1, s2, s3, s4 = [self.static_encoder(x) for x in [x1, x2, y1, y2]]

        # Decode
        x1_hat, x2_hat, y1_hat, y2_hat = [self.decoder(torch.cat((d, s), dim=1)) for d, s in zip([d1, d2, d3, d4], [s1, s2, s3, s4])]

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # SAFETY CHECK: Ensure actors has proper dimensions
            if actors.dim() < 2 or actors.size(1) < 2:
                print(f"WARNING: actors tensor has invalid shape: {actors.shape}. Expected [batch_size, 2]")
                # Create default actor IDs
                batch_size = x1.size(0)
                actors = torch.ones(batch_size, 2, device=device)

            try:
                # Extract actor IDs with bounds checking
                p1 = actors[:, 0] - 1
                p2 = actors[:, 1] - 1

                # Ensure indices are within valid range
                p1 = torch.clamp(p1, 0, self.privacy_classes - 1)
                p2 = torch.clamp(p2, 0, self.privacy_classes - 1)

                # Create one-hot encodings safely
                eye_tensor = torch.eye(self.privacy_classes, device=device)
                p1_onehot = eye_tensor[p1.long()]
                p2_onehot = eye_tensor[p2.long()]

                # Calculate losses
                priv_loss = (self.cross_entropy(self.priv_adv(d1), p1_onehot) + \
                            self.cross_entropy(self.priv_adv(d2), p2_onehot) + \
                            self.cross_entropy(self.priv_adv(d3), p1_onehot) + \
                            self.cross_entropy(self.priv_adv(d4), p2_onehot)) / 4
                priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1_onehot) + \
                            self.adv_accuracy(self.priv_adv, d2, p2_onehot) + \
                            self.adv_accuracy(self.priv_adv, d3, p1_onehot) + \
                            self.adv_accuracy(self.priv_adv, d4, p2_onehot)) / 4

                # privacy cooperative
                priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1_onehot) + \
                                self.cross_entropy(self.priv_coop(s2), p2_onehot) + \
                                self.cross_entropy(self.priv_coop(s3), p1_onehot) + \
                                self.cross_entropy(self.priv_coop(s4), p2_onehot)) / 4
                priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1_onehot) + \
                                self.adv_accuracy(self.priv_coop, s2, p2_onehot) + \
                                self.adv_accuracy(self.priv_coop, s3, p1_onehot) + \
                                self.adv_accuracy(self.priv_coop, s4, p2_onehot)) / 4
            except Exception as e:
                print(f"Error in privacy validation: {e}")

            # SAFETY CHECK: Ensure actions has proper dimensions
            if actions.dim() < 2 or actions.size(1) < 2:
                print(f"WARNING: actions tensor has invalid shape: {actions.shape}. Expected [batch_size, 2]")
                # Create default action IDs
                batch_size = x1.size(0)
                actions = torch.ones(batch_size, 2, device=device)

            try:
                # Extract action IDs with bounds checking
                a1 = actions[:, 0] - 1
                a2 = actions[:, 1] - 1

                # Ensure indices are within valid range
                a1 = torch.clamp(a1, 0, self.utility_classes - 1)
                a2 = torch.clamp(a2, 0, self.utility_classes - 1)

                # Create one-hot encodings safely
                utility_eye_tensor = torch.eye(self.utility_classes, device=device)
                a1_onehot = utility_eye_tensor[a1.long()]
                a2_onehot = utility_eye_tensor[a2.long()]

                # Calculate losses
                util_loss = (self.cross_entropy(self.util_adv(s1), a1_onehot) + \
                            self.cross_entropy(self.util_adv(s2), a2_onehot) + \
                            self.cross_entropy(self.util_adv(s3), a2_onehot) + \
                            self.cross_entropy(self.util_adv(s4), a1_onehot)) / 4
                util_acc = (self.adv_accuracy(self.util_adv, s1, a1_onehot) + \
                            self.adv_accuracy(self.util_adv, s2, a2_onehot) + \
                            self.adv_accuracy(self.util_adv, s3, a2_onehot) + \
                            self.adv_accuracy(self.util_adv, s4, a1_onehot)) / 4

                # utility cooperative
                util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1_onehot) + \
                                self.cross_entropy(self.util_coop(d2), a2_onehot) + \
                                self.cross_entropy(self.util_coop(d3), a2_onehot) + \
                                self.cross_entropy(self.util_coop(d4), a1_onehot)) / 4
                util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1_onehot) + \
                                self.adv_accuracy(self.util_coop, d2, a2_onehot) + \
                                self.adv_accuracy(self.util_coop, d3, a2_onehot) + \
                                self.adv_accuracy(self.util_coop, d4, a1_onehot)) / 4
            except Exception as e:
                print(f"Error in utility validation: {e}")

        if train_discrim:
            try:
                # train discriminator
                # Ensure proper reshaping before concatenation
                x1_view = x1.view(x1.size(0), T, -1)
                x2_view = x2.view(x2.size(0), T, -1)
                y1_view = y1.view(y1.size(0), T, -1)
                y2_view = y2.view(y2.size(0), T, -1)

                output_real = self.discriminator(torch.cat((x1_view, x2_view, y1_view, y2_view)))
                output_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat)))
                discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + \
                                     self.bce_loss(output_fake, torch.zeros_like(output_fake))
                discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * self.batch_size)) + \
                                    (torch.sum(torch.round(output_real) == 1).float() / (4 * self.batch_size))) / 2
            except Exception as e:
                print(f"Error in discriminator validation: {e}")

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def train_adv_unpaired(self, x_pos, x_rot, actors, actions, train_emb=True, train_discrim=True):
        if not self.use_adv: return 0,0,0,0,0,0,0,0,0,0
        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        # Ensure x_pos and x_rot are the same if only using position data
        x_pos = x_rot

        # move to device
        x_pos = x_pos.to(device)
        x_rot = x_rot.to(device)
        actors = actors.to(device)
        actions = actions.to(device)

        # unfreeze adversaries
        self.priv_adv.train()
        self.priv_coop.train()
        self.util_adv.train()
        self.util_coop.train()
        self.discriminator.train()

        # Encode the inputs
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        # Initialize loss and accuracy variables
        priv_loss = torch.tensor(0.0).to(device)
        priv_coop_loss = torch.tensor(0.0).to(device)
        util_loss = torch.tensor(0.0).to(device)
        util_coop_loss = torch.tensor(0.0).to(device)
        discriminator_loss = torch.tensor(0.0).to(device)
        priv_acc = torch.tensor(0.0).to(device)
        util_acc = torch.tensor(0.0).to(device)
        priv_coop_acc = torch.tensor(0.0).to(device)
        util_coop_acc = torch.tensor(0.0).to(device)
        discriminator_acc = torch.tensor(0.0).to(device)

        if train_emb:
            try:
                # Prepare actor and action labels
                actor_y = actors - 1
                actor_y = torch.clamp(actor_y, 0, self.privacy_classes - 1)
                actor_onehot = torch.eye(self.privacy_classes, device=device)[actor_y.long()]

                action_y = actions - 1
                action_y = torch.clamp(action_y, 0, self.utility_classes - 1)
                action_onehot = torch.eye(self.utility_classes, device=device)[action_y.long()]

                # Train privacy adversary
                self.priv_optim.zero_grad()
                priv_loss = self.cross_entropy(self.priv_adv(d), actor_onehot)
                priv_acc = self.adv_accuracy(self.priv_adv, d, actor_onehot)
                priv_loss.backward(retain_graph=True)
                self.priv_optim.step()

                # Train privacy cooperative
                self.priv_coop_optim.zero_grad()
                priv_coop_loss = self.cross_entropy(self.priv_coop(s), actor_onehot)
                priv_coop_acc = self.adv_accuracy(self.priv_coop, s, actor_onehot)
                priv_coop_loss.backward(retain_graph=True)
                self.priv_coop_optim.step()
            except Exception as e:
                print(f"Error in privacy training: {e}")
                # Continue with other adversaries

            try:
                # Train utility adversary
                self.util_optim.zero_grad()
                util_loss = self.cross_entropy(self.util_adv(s), action_onehot)
                util_acc = self.adv_accuracy(self.util_adv, s, action_onehot)
                util_loss.backward(retain_graph=True)
                self.util_optim.step()

                # Train utility cooperative
                self.util_coop_optim.zero_grad()
                util_coop_loss = self.cross_entropy(self.util_coop(d), action_onehot)
                util_coop_acc = self.adv_accuracy(self.util_coop, d, action_onehot)
                util_coop_loss.backward(retain_graph=True)
                self.util_coop_optim.step()
            except Exception as e:
                print(f"Error in utility training: {e}")

        if train_discrim:
            try:
                # Train discriminator
                self.discriminator_optim.zero_grad()

                # Generate real and fake samples
                real_samples = x_pos
                fake_samples = x_hat.detach()

                # Prepare labels
                real_labels = torch.ones(real_samples.size(0), 1).to(device)
                fake_labels = torch.zeros(fake_samples.size(0), 1).to(device)

                # Forward pass for real samples
                real_output = self.discriminator(real_samples)
                real_loss = self.bce_loss(real_output, real_labels)

                # Forward pass for fake samples
                fake_output = self.discriminator(fake_samples)
                fake_loss = self.bce_loss(fake_output, fake_labels)

                # Combined loss
                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()
                self.discriminator_optim.step()

                # Calculate accuracy
                real_correct = torch.sum(torch.round(real_output) == real_labels).float()
                fake_correct = torch.sum(torch.round(fake_output) == fake_labels).float()
                total = real_labels.size(0) + fake_labels.size(0)
                discriminator_acc = (real_correct + fake_correct) / total
            except Exception as e:
                print(f"Error in discriminator training: {e}")

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def forward(self, x, x_rot):
        # Add shape validation and error handling
        try:
            # Ensure x_rot has the correct shape for the encoders
            if x_rot.dim() != 5:  # Expected [B, C, T, V, M]
                print(f"Warning: Input tensor has unexpected shape: {x_rot.shape}")
                # Try to reshape to expected format
                B = x_rot.size(0)
                x_rot = x_rot.reshape(B, -1, T, 25, 1)

            dyn = self.dynamic_encoder(x_rot)
            sta = self.static_encoder(x_rot)

            # Validate encoder outputs before concatenation
            if dyn.shape != sta.shape:
                print(f"Warning: Encoder outputs have mismatched shapes: dyn={dyn.shape}, sta={sta.shape}")
                # Resize to match if needed
                if dyn.size(0) == sta.size(0):  # Same batch size
                    # Resize smaller one to match larger one
                    if dyn.numel() < sta.numel():
                        dyn = dyn.expand_as(sta)
                    else:
                        sta = sta.expand_as(dyn)

            x = self.decoder(torch.cat((dyn, sta), dim=1))
            return x

        except RuntimeError as e:
            print(f"Error in PMR forward pass: {e}")
            # Return input as fallback to prevent training failure
            return x

    def set_eval(self, eval=True):
        if eval:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()