import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    """Get available device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'