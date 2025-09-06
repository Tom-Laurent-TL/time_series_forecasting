import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        out = self.transformer_encoder(x)
        out = out[:, -1, :]  # last time step for each batch
        out = self.fc_out(out)
        return out
    
class DecompositionTimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=512):
        super().__init__()
        # Learned decomposition: trend and seasonality via conv layers
        self.trend_conv = nn.Conv1d(input_size, 1, kernel_size=5, padding=2)
        self.trend_norm = nn.BatchNorm1d(1)
        self.seasonal_conv = nn.Conv1d(input_size, 1, kernel_size=3, padding=1)
        self.seasonal_norm = nn.BatchNorm1d(1)
        # Combine decomposed features
        self.input_proj = nn.Linear(2, d_model)  # trend + seasonality
        # Time embedding
        self.time_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, 1]
        # Decomposition
        trend = self.trend_conv(x.transpose(1, 2))  # [batch, 1, seq_len]
        trend = self.trend_norm(trend)
        trend = trend.transpose(1, 2)  # [batch, seq_len, 1]
        seasonality = self.seasonal_conv(x.transpose(1, 2))  # [batch, 1, seq_len]
        seasonality = self.seasonal_norm(seasonality)
        seasonality = seasonality.transpose(1, 2)  # [batch, seq_len, 1]
        # Concatenate decomposed features
        combined = torch.cat([trend, seasonality], dim=-1)  # [batch, seq_len, 2]
        x_proj = self.input_proj(combined)  # [batch, seq_len, d_model]
        # Add time embedding
        seq_len = x_proj.size(1)
        time_indices = torch.arange(seq_len, device=x_proj.device).unsqueeze(0).repeat(x_proj.size(0), 1)  # [batch, seq_len]
        time_emb = self.time_embedding(time_indices)  # [batch, seq_len, d_model]
        x_proj = x_proj + time_emb
        out = self.transformer_encoder(x_proj)
        out = out[:, -1, :]  # last time step
        out = self.fc_out(out)
        return out
