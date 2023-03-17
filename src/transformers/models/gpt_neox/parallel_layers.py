import torch
import torch.distributed

import torch.nn.functional as F

from torch import nn


class TensorParallelColumnLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        assert out_features % self.tp_world_size == 0
        out_features = out_features // self.tp_world_size

        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

    @staticmethod
    def linear(input, weight, bias):
        return F.linear(input, weight, bias)

    def forward(self, input):
        return self.linear(input, self.weight, self.bias)


class TensorParallelRowLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        assert in_features % self.tp_world_size == 0
        in_features = in_features // self.tp_world_size

        super().__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

    @staticmethod
    def linear(input, weight, bias):
        return F.linear(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input, self.weight, self.bias)
        torch.distributed.all_reduce(out, group=self.process_group)

        return out


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.original_num_embeddings = num_embeddings

        # TODO @thomasw21 fix and remove that constraint
        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check
        if torch.any(torch.logical_or(0 > input, input >= self.original_num_embeddings)):
            raise IndexError(
                f"Input is required to be in [0, {self.original_num_embeddings}[, got min: {torch.min(input)} and max: {torch.max(input)}"
            )

        # `0` if input is in the correct interval, else `1`
        input_mask = torch.logical_or(self.min_id > input, input >= self.max_id)
        # translate for [0, self.max_id - self.min_id[
        input = input - self.min_id
        # default all out of bounds values to `0`
        input[input_mask] = 0
        out = super().forward(input)
        out[input_mask] = 0.0
        torch.distributed.all_reduce(out, group=self.process_group)
        return out
