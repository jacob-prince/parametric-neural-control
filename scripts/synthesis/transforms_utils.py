import torch
import torch.nn as nn

class SummaryFlatten(nn.Module):
    def __init__(self, summary_idxs: torch.LongTensor):
        super().__init__()
        self.register_buffer('summary_idxs', summary_idxs)  # moves with .to(device), etc.

    def forward(self, X: torch.Tensor):
        # X: (B, T, C) → returns selected summary indices flattened across time
        return X[:, self.summary_idxs, :].flatten(1)

class ExtraClsFlatten(nn.Module):
    def __init__(self, max_summary_idx: int, num_cls_tokens: int):
        super().__init__()
        self.max_summary_idx = max_summary_idx
        self.num_cls_tokens = num_cls_tokens

    def forward(self, X: torch.Tensor):
        # X: (B, T, C) → returns extra CLS tokens flattened across time
        return X[:, self.max_summary_idx:self.num_cls_tokens, :].flatten(1)

class ClsWithoutSummaryFlatten(nn.Module):
    def __init__(self, max_summary_idx: int, num_skip: int):
        super().__init__()
        self.max_summary_idx = max_summary_idx
        self.num_skip = num_skip

    def forward(self, X: torch.Tensor):
        # X: (B, T, C) → returns CLS tokens without summaries, flattened
        return X[:, self.max_summary_idx:self.num_skip, :].flatten(1)

class MaxPoolSpaceToken(nn.Module):
    def __init__(self, num_skip: int):
        super().__init__()
        self.num_skip = num_skip

    def forward(self, X: torch.Tensor):
        # X: (B, T, C) → max-pools spatial tokens, returns (B, C)
        return X[:, self.num_skip:, :].max(dim=1)[0]

class AvgPoolSpaceToken(nn.Module):
    def __init__(self, num_skip: int):
        super().__init__()
        self.num_skip = num_skip

    def forward(self, X: torch.Tensor):
        # X: (B, T, C) → mean-pools spatial tokens, returns (B, C)
        return X[:, self.num_skip:, :].mean(dim=1)
