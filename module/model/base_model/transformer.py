import torch
import torch.nn as nn
from module.model.transformer.learnable_position_encoder import LearnablePositionalEncoding


class HerculesTransformer(nn.Module):
    def __init__(self, num_layers, num_heads, d_model=768, contrastive_embedding_dim=512):
        super().__init__()
        self.d_model = d_model

        self.pos_encoder = LearnablePositionalEncoding(max_seq_len=150000, d_model=d_model)
        self.input_proj = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, contrastive_embedding_dim)
        )

    def forward(self, x):
        # x: [batch, channels, seq_len] from FeatureExtractor
        # Convert to [batch, seq_len, d_model] for Transformer
        x = x.permute(0, 2, 1)

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)

        pos_emb = self.pos_encoder(positions)
        x = x + pos_emb
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.projector(x)