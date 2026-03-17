import copy
import torch
import torch.nn as nn


class HerculesMomentumContrast(nn.Module):
    def __init__(
            self,
            base_encoder:nn.Module,
            queue_size:int = 65536,
            embedding_dim:int = 512,
            momentum_coefficient:float = 0.99,
            softmax_temperature:float = 0.07,
    ):
        super(HerculesMomentumContrast, self).__init__()
        self.momentum_coefficient = momentum_coefficient
        self.softmax_temperature = softmax_temperature
        self.queue_size = queue_size

        self.encoder_query = base_encoder
        self.encoder_key = copy.deepcopy(base_encoder)  # independent copy for momentum update

        for param_key in self.encoder_key.parameters():
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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_query.parameters(), self.encoder_key.parameters()):
            param_k.data = param_k.data * self.momentum_coefficient + param_q.data * (1. - self.momentum_coefficient)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeues old keys and enqueues new keys
        :param keys:
        :return:
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f"batch_size {batch_size} must divide queue_size {self.queue_size}"
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, signal_q, signal_k):
        q = self.encoder_query(signal_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_key(signal_k)
            k = nn.functional.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.softmax_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels
