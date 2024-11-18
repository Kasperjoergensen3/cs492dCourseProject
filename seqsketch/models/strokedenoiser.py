import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
from seqsketch.models import (
    DownSample,
    ResBlock,
    Swish,
    TimeEmbedding,
    UpSample,
    ImageEmbedding,
)
from torch.nn.utils.rnn import pack_padded_sequence


class SingleStrokeEncoder(nn.Module):
    """RNN-based encoder for a single stroke."""

    def __init__(self, d_model):
        super().__init__()
        self.rnn = nn.GRU(2, d_model, batch_first=True)

    def forward(self, stroke, mask=None):
        """
        Encode the stroke sequence.
        Args:
            stroke: Tensor of shape (B, 1, L, 2), where B is batch size, L is sequence length.
            mask: Tensor of shape (B, 1, L, 2), indicating the padding elements.
        Returns:
            Encoded stroke of shape (B, d_model).
        """
        # Compute lengths of each sequence in the batch
        if mask is not None:
            lengths = mask[..., 0].squeeze(1).sum(1)  # (B,)
        else:
            lengths = torch.tensor([stroke.size(2)] * stroke.size(0))  # (B,)

        # Pack the embedded sequence for the RNN
        print(stroke.squeeze(1).shape)
        packed_stroke = pack_padded_sequence(
            stroke.squeeze(1), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        print(packed_stroke)

        # Pass through the RNN and extract the last hidden state
        _, stroke_embed = self.rnn(packed_stroke)
        stroke_embed = stroke_embed.squeeze(0)
        return stroke_embed  # (B, d_model)
