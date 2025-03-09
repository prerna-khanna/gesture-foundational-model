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
    
class ContrastiveLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        # LSTM returns outputs and a tuple of (hidden_state, cell_state)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the final hidden state from the sequence
        features = lstm_out[:, -1, :]
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits

"""class ContrastiveTransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, num_layers=2, proj_dim=128):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Use the [CLS] token or average pooling
        features = transformer_out.mean(dim=1)
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
"""    

class ContrastiveTransformerClassifier(nn.Module): # Modified Transformer Classifier with Dropout
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, num_layers=2, proj_dim=128, dropout=0.4):
        super().__init__()
        
        # Add input normalization to stabilize training
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Add dropout after the input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # Modified transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dropout=dropout,  # Add dropout to attention and feedforward
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add dropout before classification
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Keep your projector as is
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
        # Initialize weights to prevent overfitting
        self._init_weights()
        
    def _init_weights(self):
        # More conservative initialization to prevent overfitting
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x, return_features=False):
        # Normalize input
        x = self.input_norm(x)
        
        # Project input to hidden dimension with dropout
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Apply feature averaging with dropout
        features = transformer_out.mean(dim=1)
        features = self.feature_dropout(features)
        
        # Apply classifier
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
    

class ContrastiveBiGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        # Using bidirectional GRU with half the hidden dims per direction
        # This ensures the total output dimension remains hidden_dim
        self.bigru = nn.GRU(
            input_dim, 
            hidden_dim // 2,  # Half the dimension for each direction
            batch_first=True,
            bidirectional=True  # Enable bidirectional processing
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        # BiGRU returns outputs and hidden states from both directions
        # outputs shape: [batch, seq_len, hidden_dim]
        bigru_out, _ = self.bigru(x)
        
        # Take the final hidden state from the sequence
        # This automatically concatenates both directions
        features = bigru_out[:, -1, :]
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits

class ContrastiveBiLSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Attention mechanism
        att_weights = self.attention(lstm_out)  # [batch, seq, 1]
        att_weights = F.softmax(att_weights, dim=1)
        features = torch.sum(lstm_out * att_weights, dim=1)  # [batch, hidden]
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
    
class ContrastiveTCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, kernel_size=3, proj_dim=128):
        super().__init__()
        
        # Build TCN layers
        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    input_dim if _ == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    padding=(kernel_size-1) * dilation // 2,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            dilation *= 2
        
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        # Transpose for TCN: [batch, seq, features] -> [batch, features, seq]
        x = x.transpose(1, 2)
        
        # Apply TCN
        tcn_out = self.tcn(x)
        
        # Global pooling
        features = F.adaptive_max_pool1d(tcn_out, 1).squeeze(-1)
        
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
    
class ContrastiveSVMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128):
        super().__init__()
        # Feature extractor - using GRU to process the time series data
        self.feature_extractor = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # SVM-like classifier with hinge loss
        # We implement this as a linear layer without bias for the SVM-like behavior
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)
        
        # Projector for contrastive learning
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
        
    def forward(self, x, return_features=False):
        # Extract features from sequence data
        feature_output, _ = self.feature_extractor(x)
        features = feature_output[:, -1, :]  # Take the final hidden state
        
        # Linear SVM-like classification (without softmax)
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits