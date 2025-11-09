import torch.nn as nn
from omnilearned.layers import (
    AttBlock,
    DynamicTanh,
)


class MPM_head(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        # num_tokens=0,
        codebook_size_continuous=5000,
        codebook_size_pid=None,
    ):
        super().__init__()
        # self.num_tokens = num_tokens
        self.codebook_size = codebook_size_continuous
        self.codebook_size_pid = codebook_size_pid

        self.blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=0,
                    skip=False,
                    use_int=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.out_continuous = nn.Linear(hidden_size, codebook_size_continuous)

        if codebook_size_pid is None:
            self.out_pid = None
        else:
            self.out_pid = nn.Linear(hidden_size, codebook_size_pid)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, mask):
        for ib, blk in enumerate(self.blocks):
            x = blk(x, mask=mask)

        logits_continuous = self.out_continuous(x) * mask

        if self.codebook_size_pid is None:
            logits_pid = None
        else:
            logits_pid = self.out_pid(x) * mask

        return logits_continuous, logits_pid
