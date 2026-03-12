import torch
import torch.nn as nn


class HerculesMomentumContrast(nn.Module):
    def __init__(
            self,
            base_encoder:nn.Sequential,
            queue_size:int = 65536,
            embedding_dim:int = 512,
            momentum:float = 0.99,
            softmax_temperature:float = 0.07,
    ):
        super(HerculesMomentumContrast, self).__init__()
        self.base_encoder = base_encoder
        self.momentum = momentum
        self.softmax_temperature = softmax_temperature

        self.encoder_query = base_encoder
        self.encoder_key = base_encoder

        for param_query, param_key in zip(self.encoder_query.parameters(), self.encoder_key.parameters()):
            param_key.data.copy_(param_query.data)
            param_key.requires_grad_(False)

        self.register_buffer(
            "queue",
            torch.randn(embedding_dim, queue_size)
        ) # create a random queue negative samples
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long)
        )
        # TODO
