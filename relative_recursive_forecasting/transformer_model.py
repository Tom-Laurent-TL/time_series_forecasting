# RecursiveTimeSeriesTransformer for time series forecasting

import torch
import torch.nn as nn
import math
import numpy as np
import pywt  # PyWavelets for wavelet transforms

class RecursiveTimeSeriesTransformer(nn.Module):
	def __init__(self, seq_length, forecast_steps, d_model=128, nhead=8, num_layers=3, dropout=0.1, time_feat_size=7, time_emb_dim=8):
		super().__init__()
		self.seq_length = seq_length
		self.forecast_steps = forecast_steps
		self.d_model = d_model
		self.time_emb_dim = time_emb_dim

		self.input_proj = nn.Linear(1, d_model)
		self.time_embedding = nn.Embedding(time_feat_size, time_emb_dim)
		self.concat_proj = nn.Linear(d_model + time_emb_dim, d_model)

		# TCN encoder for non-stationary data
		self.tcn_encoder = nn.Sequential(
			nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.LayerNorm([d_model, seq_length])
		)

		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.layer_norm = nn.LayerNorm(d_model)
		self.attn_pool = nn.Linear(d_model, 1)
		self.output_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.LayerNorm(d_model),
			nn.Linear(d_model, 1)
		)

		# Trend/seasonality decomposition block
		self.decomposition_block = nn.Sequential(
			nn.Linear(seq_length, seq_length),
			nn.ReLU(),
			nn.Linear(seq_length, seq_length)
		)


	def wavelet_transform(self, x, wavelet='db1', level=1):
		# x: (batch_size, seq_length)
		wt = []
		for seq in x.cpu().numpy():
			coeffs = pywt.wavedec(seq, wavelet, level=level)
			# Concatenate all coefficients for compression
			wt_seq = np.concatenate([c for c in coeffs])
			wt.append(wt_seq)
		wt = torch.tensor(wt, dtype=x.dtype, device=x.device)
		return wt

	def attention_pooling(self, memory):
		# memory: (batch_size, seq_length, d_model)
		attn_weights = torch.softmax(self.attn_pool(memory), dim=1)  # (batch_size, seq_length, 1)
		pooled = torch.sum(memory * attn_weights, dim=1)  # (batch_size, d_model)
		return pooled

	def forward(self, x, time_feat_idx=None):
		# x: (batch_size, seq_length)
		# time_feat_idx: (batch_size, seq_length) integer indices for time features (e.g., day of week)
		batch_size = x.size(0)
		device = x.device
		seq_length = self.seq_length
		d_model = self.d_model

		current_seq = x.clone()  # (batch_size, seq_length)
		preds = []

		for step in range(self.forecast_steps):
			# Decomposition block for trend/seasonality
			decomposed = self.decomposition_block(current_seq)
			inp_seq = current_seq.unsqueeze(-1)  # (batch_size, seq_length, 1)
			inp_seq = self.input_proj(inp_seq)  # (batch_size, seq_length, d_model)

			if time_feat_idx is not None:
				# Use time feature indices for embedding
				if time_feat_idx.dim() == 2:
					time_emb = self.time_embedding(time_feat_idx)  # (batch_size, seq_length, time_emb_dim)
				else:
					# If 1D, expand to match batch and seq_length
					time_emb = self.time_embedding(time_feat_idx.unsqueeze(0).expand(batch_size, seq_length))
				inp_seq = torch.cat([inp_seq, time_emb], dim=-1)  # (batch_size, seq_length, d_model + time_emb_dim)
				inp_seq = self.concat_proj(inp_seq)  # (batch_size, seq_length, d_model)

			# Removed sinusoidal positional encoding addition

			# TCN encoder for non-stationary data
			tcn_inp = inp_seq.transpose(1, 2)  # (batch_size, d_model, seq_length)
			tcn_out = self.tcn_encoder(tcn_inp)  # (batch_size, d_model, seq_length)
			tcn_out = tcn_out.transpose(1, 2)  # (batch_size, seq_length, d_model)

			# Transformer encoder with residual and layer norm
			memory = self.transformer_encoder(tcn_out)
			memory = self.layer_norm(memory + tcn_out)
			# Attention pooling
			pooled = self.attention_pooling(memory)
			# Output head with residual and layer norm
			out = self.output_head(pooled + pooled)
			preds.append(out)
			current_seq = torch.cat([current_seq[:, 1:], out.squeeze(-1).unsqueeze(1)], dim=1)
			if time_feat_idx is not None:
				# Shift time feature indices for next step
				time_feat_idx = torch.cat([time_feat_idx[:, 1:], time_feat_idx[:, -1:].clone()], dim=1)

		preds = torch.cat(preds, dim=1)
		return preds
