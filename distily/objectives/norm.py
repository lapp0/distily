import torch.nn as nn


class BatchNorm1d4d(nn.Module):
    """
    Custom BatchNorm that accepts a 4D tensor of shape
    (batch size, num layers, sequence length, feature size)
    and applies batch normalization
    """
    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.batchnorm1d = nn.BatchNorm1d(num_features, eps=eps, **kwargs)

    def forward(self, x):
        """
        Calculate norm across batch, features, sequence elements
        num norms = feature size
        """
        batch_size, layers, sequence_length, feature_size = x.shape
        x_flattened = x.view(batch_size * layers * sequence_length, feature_size)
        x_flattened_norm = self.batchnorm1d(x_flattened)

        # Reshape back to original dimensions
        x_norm = x_flattened_norm.view(batch_size, layers, sequence_length, feature_size)
        return x_norm


class LayerNorm1d4d(nn.LayerNorm):
    """
    Calculate layernorm across 4D tensor of shape
    (batch size, num layers, sequence length, feature size)

    Calculate norm across batch, layers, sequence elements
    Calculate norm across features
    """
    pass
