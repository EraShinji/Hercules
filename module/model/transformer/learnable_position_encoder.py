import torch.nn as nn
class LearnablePositionalEncoding(nn.Module):
   def __init__(self, max_seq_len, d_model):
       super().__init__()
       self.pos_embedding = nn.Embedding(max_seq_len, d_model)
   def forward(self, positions):
       return self.pos_embedding(positions)