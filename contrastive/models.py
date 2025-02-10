import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return F.normalize(x, dim=1)


class ContrastiveGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        gru_out, _ = self.gru(x)
        features = gru_out[:, -1, :]
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits