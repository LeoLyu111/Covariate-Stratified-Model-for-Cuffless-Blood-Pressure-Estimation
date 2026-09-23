import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class BPTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=4, num_encoder_layers=2, dim_feedforward=64, 
                 max_length=100000, num_features=0, input_signals='ppg_ecg'):
        super(BPTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        self.num_features = num_features
        self.input_signals = input_signals
        
        # Determine number of input channels based on signal type
        if input_signals == 'ppg_ecg':
            input_channels = 2  # PPG + ECG
        elif input_signals == 'ppg':
            input_channels = 1  # PPG only
        else:
            raise ValueError("input_signals must be 'ppg_ecg' or 'ppg'")
        
        # Input projection for signals
        self.input_projection = nn.Linear(input_channels, d_model)

        # Learnable token used to aggregate information from the full sequence.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        # Reserve one additional position for the CLS token.
        self.pos_encoder = PositionalEncoding(d_model, max_length + 1)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Feature processing if features are provided
        if num_features > 0:
            self.feature_processor = nn.Sequential(
                nn.Linear(num_features, d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        
        # Final regression heads
        input_dim = d_model + (d_model if num_features > 0 else 0)
        self.sbp_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        self.dbp_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, signals, features=None):
        # signals shape: (batch_size, num_channels, max_length)
        batch_size = signals.size(0)
        
        # Transpose to (batch_size, max_length, num_channels)
        signals = signals.transpose(1, 2)
        
        # Project to d_model
        x = self.input_projection(signals)  # (batch_size, max_length, d_model)

        # Prepend the same learnable CLS token to every sample. Through
        # self-attention it learns to aggregate information from all time steps.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, max_length + 1, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (max_length + 1, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, max_length + 1, d_model)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # (batch_size, max_length + 1, d_model)
        
        # Use the contextualized CLS representation for regression.
        x = x[:, 0, :]  # (batch_size, d_model)
        
        # Process features if available
        if features is not None and self.num_features > 0:
            feat_processed = self.feature_processor(features)  # (batch_size, d_model)
            x = torch.cat([x, feat_processed], dim=1)  # (batch_size, 2*d_model)
        
        # Predict BP values
        sbp_pred = self.sbp_head(x)
        dbp_pred = self.dbp_head(x)
        
        return sbp_pred, dbp_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
