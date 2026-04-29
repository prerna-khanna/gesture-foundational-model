import torch
import torch.nn.functional as F
import random
import math
import numpy as np


class GestureAugmenter:
    def __init__(self, jitter_scale=0.1, speed_scale=0.2, rotation_scale=0.1):
        """
        Initialize augmentation parameters for gesture data
        Args:
            jitter_scale: Controls the magnitude of random noise
            speed_scale: Controls the range of speed variation
            rotation_scale: Controls the degree of rotational perturbation
        """
        self.jitter_scale = jitter_scale
        self.speed_scale = speed_scale
        self.rotation_scale = rotation_scale
    
    def augment(self, x):
        """
        Create augmented view of the gesture data
        Args:
            x: Input numpy array of shape [sequence_length, features]
        Returns:
            Augmented numpy array of the same shape
        """
        # Make a copy of input array instead of using tensor clone
        x = np.copy(x)
        return self._apply_augmentation(x)
    
    def _apply_augmentation(self, x):
        """Apply augmentations in random order to maintain variety"""
        augmentations = [self._add_jitter, self._vary_speed, self._rotate_signal]
        random.shuffle(augmentations)
        
        for aug in augmentations:
            x = aug(x)
        return x
    
    def _add_jitter(self, x):
        """Add random noise scaled by signal magnitude"""
        noise = np.random.randn(*x.shape) * self.jitter_scale * np.std(x)
        return x + noise
    
    def _vary_speed(self, x):
        """Vary the speed of the gesture through interpolation"""
        seq_len, features = x.shape
        speed_factor = 1.0 + (np.random.rand() - 0.5) * 2 * self.speed_scale
        new_len = int(seq_len * speed_factor)
        
        # Ensure reasonable sequence length
        new_len = max(seq_len // 2, min(seq_len * 2, new_len))
        
        # Create time vectors for interpolation
        orig_time = np.linspace(0, 1, seq_len)
        new_time = np.linspace(0, 1, new_len)
        
        # Interpolate each feature dimension
        x_new = np.zeros((new_len, features))
        for i in range(features):
            x_new[:, i] = np.interp(new_time, orig_time, x[:, i])
        
        # Interpolate back to original length
        x_final = np.zeros((seq_len, features))
        new_time = np.linspace(0, 1, new_len)
        orig_time = np.linspace(0, 1, seq_len)
        for i in range(features):
            x_final[:, i] = np.interp(orig_time, new_time, x_new[:, i])
            
        return x_final
    
    def _rotate_signal(self, x):
        """Apply small rotational perturbation to the signal"""
        angle = (np.random.rand() - 0.5) * 2 * self.rotation_scale * np.pi
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Create rotation matrix
        rot_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        # Apply rotation to first two dimensions
        x_rotated = np.dot(x[:, :2], rot_matrix.T)
        x[:, :2] = x_rotated
        return x