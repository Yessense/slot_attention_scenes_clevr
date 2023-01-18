import torch
from torch import nn
from torch.nn import init
import pytorch_lightning as pl


class SlotAttention(pl.LightningModule):
    """Slot Attention Module."""

    def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
                 epsilon=1e-8):
        """
        :param num_iterations: Number of iterations
        :param num_slots: Number of slots.
        :param slot_size:  Dimensionality of slot feature vectors
        :param mlp_hidden_size: Hidden layer size of MLP
        :param epsilon: Offset for attention coefficients before normalization
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_state = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.slot_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        self.slots_mu = nn.Parameter(data=torch.randn(1, 1, self.slot_size))
        init.xavier_uniform_(self.slots_mu)

        self.slots_log_sigma = nn.Parameter(data=torch.randn(1, 1, self.slot_size))
        init.xavier_uniform_(self.slots_log_sigma)

        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        self.gru = nn.GRUCell(self.slot_size, self.slot_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_state),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden_state, self.slot_size)
        )

    def forward(self, inputs):
        batch_size, num_inputs, inputs_size = inputs.shape
        # inputs -> (batch_size, num_inputs, inputs_size)
        inputs = self.norm_inputs(inputs)

        k = self.project_k(inputs)
        v = self.project_v(inputs)
        # k, v -> (batch_size, num_inputs, slot_size)

        slots = self.slots_mu + self.slots_log_sigma * torch.randn((batch_size, self.num_slots, self.slot_size), device=self.device)
        # slots -> (batch_size, num_slots, slot_size)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q *= self.slot_size ** -0.5  # Normalization
            # q -> (batch_size, num_slots, slot_size)

            # Attention
            attn_logits = torch.einsum('bid,bsd->bis', k, q)
            attn = attn_logits.softmax(dim=-1)
            # attn -> (batch_size, num_inputs, num_slots)

            # Weighted mean
            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=-1, keepdims=True)
            updates = torch.einsum('bis,bid->bsd', attn, v)

            # Slot update
            slots = self.gru(updates.reshape(-1, inputs_size), slots_prev.reshape(-1, inputs_size))

            slots = slots.reshape(batch_size, -1, inputs_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots


if __name__ == '__main__':
    sa = SlotAttention(num_iterations=2,
                       num_slots=7,
                       slot_size=128,
                       mlp_hidden_size=128,
                       epsilon=1e-8)

    print("Done")
