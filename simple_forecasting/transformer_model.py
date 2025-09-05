import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_length)
        src = src.unsqueeze(-1)  # (batch_size, seq_length, 1)
        src = self.input_linear(src)  # (batch_size, seq_length, d_model)
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        out = self.transformer_encoder(src)  # (seq_length, batch_size, d_model)
        out = out[-1]  # (batch_size, d_model)
        out = self.output_linear(out)  # (batch_size, 1)
        return out.squeeze(-1)
