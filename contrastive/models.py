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

class ContrastiveCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, feature_dim=128, num_layers=3, kernel_size=3, proj_dim=128):
        super().__init__()
        
        # Store the feature dimension for projection consistency
        self.feature_dim = feature_dim
        
        # CHANGE 1: Reduce model capacity by decreasing channel growth rate
        channel_sizes = [input_dim, hidden_dim//2, hidden_dim, feature_dim]
        
        # Conv layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # Add main conv block
            self.conv_layers.append(
                nn.Conv1d(channel_sizes[i], channel_sizes[i+1], 
                         kernel_size=kernel_size, padding=kernel_size//2)
            )
            # CHANGE 2: Use Layer Normalization instead of Batch Normalization
            self.bn_layers.append(nn.GroupNorm(8, channel_sizes[i+1]))
            
            # Add downsampling layer (pool every other layer)
            if i % 2 == 1:
                self.downsample_layers.append(nn.MaxPool1d(2))
            else:
                self.downsample_layers.append(nn.Identity())
        
        # Adaptive pooling to get fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(channel_sizes[-1])
        
        # CHANGE 3: Add an additional bottleneck layer before classification
        self.bottleneck = nn.Sequential(
            nn.Linear(channel_sizes[-1], channel_sizes[-1]//2),
            nn.ReLU(),
            nn.LayerNorm(channel_sizes[-1]//2),
            nn.Linear(channel_sizes[-1]//2, channel_sizes[-1])
        )
        
        # Classification head
        self.classifier = nn.Linear(channel_sizes[-1], num_classes)
        
        # Projection head for contrastive learning
        self.projector = ProjectionHead(channel_sizes[-1], hidden_dim, proj_dim)
        
        # CHANGE 4: Increase dropout for stronger regularization
        self.dropout = nn.Dropout(0.5)
        
        # CHANGE 5: Add spatial dropout for feature maps
        self.spatial_dropout = nn.Dropout2d(0.3)
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, features = x.shape
        
        # Transpose for 1D convolution (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN layers with residual connections where possible
        for i, (conv, bn, downsample) in enumerate(zip(self.conv_layers, self.bn_layers, self.downsample_layers)):
            # Store input for residual connection
            residual = x if x.size(1) == conv.out_channels else None
            
            # Apply convolution
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            
            # CHANGE 6: Apply spatial dropout to feature maps
            if i < len(self.conv_layers) - 1:  # Skip last layer
                x = self.spatial_dropout(x.unsqueeze(3)).squeeze(3)
            
            # Apply residual connection if available
            if residual is not None:
                x = x + residual
            
            # Apply downsampling and dropout
            x = downsample(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        features = x.view(batch_size, -1)
        
        # Normalize features
        features = self.feature_norm(features)
        
        # CHANGE 7: Apply bottleneck with skip connection for better feature learning
        features = features + self.bottleneck(features)
        
        # Apply dropout before final classification
        features = self.dropout(features)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits
    
    

class ContrastiveDeepSenseClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, proj_dim=128, num_filter=8):
        super().__init__()
        
        # Assuming input_dim is divisible by 3 (number of channels per sensor)
        self.sensor_num = input_dim // 3
        self.hidden_dim = hidden_dim
        
        # Individual sensor processing
        for i in range(self.sensor_num):
            self.__setattr__(f'conv{i}_1', nn.Conv2d(1, num_filter, (2, 3)))
            self.__setattr__(f'conv{i}_2', nn.Conv2d(num_filter, num_filter, (3, 1)))
            self.__setattr__(f'conv{i}_3', nn.Conv2d(num_filter, num_filter, (2, 1)))
            self.__setattr__(f'bn{i}_1', nn.BatchNorm2d(num_filter))
            self.__setattr__(f'bn{i}_2', nn.BatchNorm2d(num_filter))
            self.__setattr__(f'bn{i}_3', nn.BatchNorm2d(num_filter))
        
        # Cross-sensor processing
        self.conv_merge1 = nn.Conv2d(1, num_filter, (2, self.sensor_num))
        self.bn_merge1 = nn.BatchNorm2d(num_filter)
        self.conv_merge2 = nn.Conv2d(num_filter, num_filter, (3, 1))
        self.bn_merge2 = nn.BatchNorm2d(num_filter)
        self.conv_merge3 = nn.Conv2d(num_filter, hidden_dim // 8, (2, 1))
        self.bn_merge3 = nn.BatchNorm2d(hidden_dim // 8)
        
        # Adaptive pooling for different sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Feature layer will be initialized in the first forward pass
        self.feature_layer = None
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Projection head for contrastive learning
        self.projector = ProjectionHead(hidden_dim, hidden_dim*2, proj_dim)
    
    def forward(self, x, return_features=False):
        batch_size, seq_len, feature_dim = x.shape
        
        # Reshape input to separate the sensors
        h = x.view(batch_size, seq_len, self.sensor_num, 3)
        
        # Process each sensor separately
        sensor_features = []
        for i in range(self.sensor_num):
            # Extract this sensor's data
            sensor_data = h[:, :, i, :]
            sensor_data = sensor_data.unsqueeze(1)  # Add channel dimension
            
            # Apply convolutional layers
            conv1 = self.__getattr__(f'conv{i}_1')
            bn1 = self.__getattr__(f'bn{i}_1')
            t = bn1(F.relu(conv1(sensor_data)))
            
            conv2 = self.__getattr__(f'conv{i}_2')
            bn2 = self.__getattr__(f'bn{i}_2')
            t = bn2(F.relu(conv2(t)))
            
            conv3 = self.__getattr__(f'conv{i}_3')
            bn3 = self.__getattr__(f'bn{i}_3')
            t = bn3(F.relu(conv3(t)))
            
            # Store features for this sensor
            sensor_features.append(t)
        
        # Concatenate all sensor features
        concat_features = torch.cat([self.flatten(f).unsqueeze(2) for f in sensor_features], dim=2)
        concat_features = concat_features.unsqueeze(1)  # Add channel dimension
        
        # Cross-sensor processing
        merge = self.bn_merge1(F.relu(self.conv_merge1(concat_features)))
        merge = self.bn_merge2(F.relu(self.conv_merge2(merge)))
        merge = self.bn_merge3(F.relu(self.conv_merge3(merge)))
        
        # Use adaptive pooling to get a fixed size output
        merge = self.adaptive_pool(merge)
        features = self.flatten(merge)
        
        # Initialize the feature layer if it doesn't exist
        if self.feature_layer is None:
            feature_size = features.shape[1]
            self.feature_layer = nn.Linear(feature_size, self.hidden_dim).to(features.device)
            print(f"Initialized feature layer with input size {feature_size}, output size {self.hidden_dim}")
        
        # Process features
        features = F.relu(self.feature_layer(features))
        features = self.feature_norm(features)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            projected = self.projector(features)
            return logits, features, projected
        
        return logits


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

class ContrastiveTransformerClassifier(nn.Module):
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


"""class ContrastiveTransformerClassifier(nn.Module): # Modified Transformer Classifier with Dropout
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
        
        return logits"""
    

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