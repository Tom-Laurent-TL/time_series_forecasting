# RecursiveTimeSeriesTransformer for time series forecasting

import torch
import torch.nn as nn
import math

class RecursiveTimeSeriesTransformer(nn.Module):
	def __init__(self, seq_length, forecast_steps, d_model=128, nhead=8, num_layers=3, dropout=0.1):
		super().__init__()
		self.seq_length = seq_length
		self.forecast_steps = forecast_steps
		self.d_model = d_model

		self.input_proj = nn.Linear(1, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.output_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, 1)
		)

	def positional_encoding(self, seq_len, d_model, device):
		pe = torch.zeros(seq_len, d_model, device=device)
		position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe

	def forward(self, x):
		# x: (batch_size, seq_length)
		batch_size = x.size(0)
		device = x.device
		seq_length = self.seq_length
		d_model = self.d_model

		# Start with the initial sequence
		current_seq = x.clone()  # (batch_size, seq_length)
		preds = []

		for _ in range(self.forecast_steps):
			# Prepare input for transformer
			inp_seq = current_seq.unsqueeze(-1)  # (batch_size, seq_length, 1)
			inp_seq = self.input_proj(inp_seq)  # (batch_size, seq_length, d_model)
			pe = self.positional_encoding(seq_length, d_model, device)
			inp_seq = inp_seq + pe.unsqueeze(0)
			memory = self.transformer_encoder(inp_seq)  # (batch_size, seq_length, d_model)
			last_hidden = memory[:, -1, :]  # (batch_size, d_model)
			out = self.output_head(last_hidden)  # (batch_size, 1)
			preds.append(out)
			# Append prediction and remove the first state to keep context size fixed
			current_seq = torch.cat([current_seq[:, 1:], out.squeeze(-1).unsqueeze(1)], dim=1)  # (batch_size, seq_length)

		preds = torch.cat(preds, dim=1)  # (batch_size, forecast_steps)
		return preds
