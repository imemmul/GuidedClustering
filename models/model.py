import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F

class SSLResNet(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )
        
        # Add prediction head
        self.prediction = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        features = self.get_features(x)
        z = self.projection(features)
        p = self.prediction(z)
        return z, p

    def get_features(self, x):
        return self.backbone(x)

def nt_xent_loss(z1, z2, temperature=0.1):
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    
    similarity_matrix = torch.matmul(z1_norm, z2_norm.T) / temperature
    
    positives = torch.diag(similarity_matrix)
    
    exp_sim = torch.exp(similarity_matrix)
    
    denominator = exp_sim.sum(dim=1) - torch.exp(positives)
    
    loss = -torch.mean(positives - torch.log(denominator))
    return loss