import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainLoss(nn.Module):
    """
    Loss function that combines time domain MSE with frequency domain loss.
    
    This is particularly useful for IMU data where frequency characteristics
    are important for capturing periodic motion patterns.
    """
    def __init__(self, alpha=0.3, reduction='none'):
        """
        Args:
            alpha: Weight of frequency domain loss (0-1). 
                  1-alpha will be the weight of time domain loss.
            reduction: 'none', 'mean', or 'sum' - how to reduce the loss.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.time_criterion = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        """
        Calculate combined loss in time and frequency domains.
        
        Args:
            pred: Predicted values [batch_size, seq_len, features]
            target: Target values [batch_size, seq_len, features]
            
        Returns:
            Combined loss
        """
        # Time domain loss
        time_loss = self.time_criterion(pred, target)
        
        # Compute FFT along the sequence dimension (dim=1)
        # Move sequence dimension to last position for FFT
        pred_permuted = pred.permute(0, 2, 1)  # [batch, features, seq_len]
        target_permuted = target.permute(0, 2, 1)  # [batch, features, seq_len]
        
        # Compute real FFT
        pred_fft = torch.fft.rfft(pred_permuted)
        target_fft = torch.fft.rfft(target_permuted)
        
        # Calculate magnitude of FFT components
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # Calculate MSE in frequency domain
        freq_loss = F.mse_loss(pred_magnitude, target_magnitude, reduction='none')
        
        # Pad freq_loss to match time_loss dimensions
        batch_size, feature_dim, freq_dim = freq_loss.shape
        seq_len = time_loss.shape[1]
        
        # Create a padded tensor initialized with zeros
        padded_freq_loss = torch.zeros(batch_size, seq_len, feature_dim, device=freq_loss.device)
        
        # Fill in the available frequency components
        padded_freq_loss[:, :freq_dim, :] = freq_loss.permute(0, 2, 1)
        
        # Combine losses
        combined_loss = (1 - self.alpha) * time_loss + self.alpha * padded_freq_loss
        
        # Apply reduction if needed
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:  # 'none'
            return combined_loss