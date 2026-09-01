
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from einops import repeat
class SSM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            modal_num=3,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.in_proj_bcd = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if self.d_conv == 1:
            padding = 0
        else:
            padding = (self.d_conv - 1) // 2

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            padding=padding,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.D = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.out_proj = self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor, x_bcd: torch.Tensor):

        B, C, L = x.shape
        B, C, L = x_bcd.shape

        K = 2

        x_tmp = torch.unsqueeze(x,dim=1)
        x_bcd_tmp = torch.unsqueeze(x_bcd,dim=1)

        xs = torch.cat([x_tmp, torch.flip(x_tmp, dims=[-1])], dim=1)
        xs_bcd = torch.cat([x_bcd_tmp, torch.flip(x_bcd_tmp, dims=[-1])], dim=1)


        t = xs_bcd.view(B, K, -1, L)
        z = self.x_proj_weight

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_bcd.view(B, K, -1, L), self.x_proj_weight)




        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)


        dts = torch.einsum("b k r l, k d r -> b k d l", dts.contiguous().view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.D.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)


        #########################################
        y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert y.dtype == torch.float

        y_forward = y[:, 0]

        y_reflip = torch.flip(y[:, 1], dims=[-1])


        return y_forward + y_reflip

    def forward(self, x_in: torch.Tensor, x_bcd: torch.Tensor,  **kwargs):

        #x_in (l, b, c)
        #x_bcd (l, b, c)

        x_in = x_in.permute(1, 0, 2) # (b, l, c)
        x_bcd = x_bcd.permute(1, 0, 2) # (b, l, c)

        Bs, Ls,  Cs = x_in.shape

        xz = self.in_proj(x_in)

        x_bcd = self.in_proj_bcd(x_bcd)

        x, z = xz.chunk(2, dim=-1)  # (b, l, c)

        x = x.permute(0, 2, 1) # (b, c, l)
        x = self.act(self.conv1d(x))

        x_bcd = x_bcd.permute(0, 2, 1)  # (b, l, c)

        y = self.forward_core(x, x_bcd)

        y = y.permute(0, 2, 1) # (b, l, c)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = x_in + self.out_proj(y)

        return out.permute(1, 0, 2)


class CrossSS2DLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.choker = nn.Linear(in_features=2 * d_model, out_features=d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.SSM = SSM(d_model=d_model)

    def forward(self,
                tgt,
                memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        x_in = tgt

        # concatenate the positional embedding with the content feature,
        # instead of direct addition
        cross_attn_q = torch.cat((tgt, query_pos + pos[memory.shape[0]:]),
                                 dim=-1)
        cross_attn_k = torch.cat((memory, pos[:memory.shape[0]]), dim=-1)

        tgt2 = self.cross_attn(
            query=cross_attn_q,
            key=cross_attn_k,
            value=memory,
            key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout1(self.choker(tgt2))
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt_out = self.SSM(x_in, tgt)

        return tgt_out

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    output = torch.randn(size=(17, 16, 256)).to(device)
    memory = torch.randn(size=(49, 16, 256)).to(device)
    tgt_key_padding_mask = torch.zeros(size=(16, 17)).to(device).bool()
    memory_key_padding_mask = torch.zeros(size=(16, 49)).to(device).bool()
    pos = torch.randn(size=(66, 16, 256)).to(device)

    initial_proposals = torch.randn(size=(16, 17, 2)).to(device)

    query_pos = torch.randn(size=(17, 16, 256)).to(device)

    decoder_layer = CrossSS2DLayer(d_model=256, nhead=8).to(device)

    y = decoder_layer(output, memory, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    print(y.shape)


    # x = torch.randn(size=(17, 16, 256)).to(device)
    # x_bcd = torch.randn(size=(17, 16, 256)).to(device)
    #
    # ss2d = SSM(d_model=256).to(device)
    #
    # y = ss2d(x, x_bcd)
    #
    # print(y.shape)

