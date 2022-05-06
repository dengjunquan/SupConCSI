from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8,   5, stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.Conv2d(8, 16,  3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.Tanh()
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(20000, 256),
            nn.Tanh(), 
            nn.Linear(256, encoded_space_dim)
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DMEncoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8,   5, stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.Conv2d(8, 16,  3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.Tanh()
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(20000, 256),
            nn.Tanh(), 
            nn.Linear(256, 2),
        #nn.Tanh(), 
        #nn.Linear(encoded_space_dim, 2),
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
