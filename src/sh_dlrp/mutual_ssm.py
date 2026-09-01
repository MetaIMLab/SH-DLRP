
import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from einops import repeat
class MutualSSM(nn.Module):
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

        self.in_projs = nn.ModuleList([nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs) for i in range(4)])



        if self.d_conv == 1:
            padding = 0
        else:
            padding = (self.d_conv - 1) // 2

        self.conv1ds = nn.ModuleList([nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            padding=padding,
            **factory_kwargs,
        ) for _ in range(4)])

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias_1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        self.dt_projs_bias_2 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        self.dt_projs_bias_3 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        self.dt_projs_bias_4 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))# (K=2, inner)
        del self.dt_projs

        self.A_logs_1 = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.A_logs_2 = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.A_logs_3 = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.A_logs_4 = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        # (K=2, D, N)
        self.Ds_1 = self.D_init(self.d_inner, copies=1, merge=True)
        self.Ds_2 = self.D_init(self.d_inner, copies=1, merge=True)
        self.Ds_3 = self.D_init(self.d_inner, copies=1, merge=True)
        self.Ds_4 = self.D_init(self.d_inner, copies=1, merge=True)# (K=2, D, N)

        self.out_proj = self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.fuse = nn.Conv1d(
            in_channels=self.d_model * self.expand * 4,
            out_channels=self.d_model * self.expand,
            bias=conv_bias,
            kernel_size=1,
            padding=0,
            **factory_kwargs,
        )

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


    def forward_core(self, x1, x2, x3, x4):

        B, C, L = x1.shape


        K = 1

        x1_s = torch.unsqueeze(x1, dim=1)
        x2_s = torch.unsqueeze(x2, dim=1)
        x3_s = torch.unsqueeze(x3, dim=1)
        x4_s = torch.unsqueeze(x4, dim=1)




        x_fuse = torch.unsqueeze(self.fuse(torch.cat([x1, x2, x3, x4], dim=1)), dim=1)




        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse.view(B, K, -1, L), self.x_proj_weight)




        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)




        dts = torch.einsum("b k r l, k d r -> b k d l", dts.contiguous().view(B, K, -1, L), self.dt_projs_weight)

        x1_s = x1_s.float().view(B, -1, L)  # (b, k * d, l)
        x2_s = x2_s.float().view(B, -1, L)  # (b, k * d, l)
        x3_s = x3_s.float().view(B, -1, L)  # (b, k * d, l)
        x4_s = x4_s.float().view(B, -1, L)  # (b, k * d, l)


        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)


        Ds_1, Ds_2, Ds_3, Ds_4 = self.Ds_1.float().view(-1), self.Ds_2.float().view(-1), self.Ds_3.float().view(-1), self.Ds_4.float().view(-1)  # (k * d)
        As_1, As_2, As_3, As_4= -torch.exp(self.A_logs_1.float()).view(-1, self.d_state), -torch.exp(self.A_logs_2.float()).view(-1, self.d_state), -torch.exp(self.A_logs_3.float()).view(-1, self.d_state), -torch.exp(self.A_logs_4.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias_1, dt_projs_bias_2, dt_projs_bias_3, dt_projs_bias_4 = self.dt_projs_bias_1.float().view(-1), self.dt_projs_bias_2.float().view(-1), self.dt_projs_bias_3.float().view(-1), self.dt_projs_bias_4.float().view(-1)  # (k * d)


        #########################################
        y_1 = self.selective_scan(
            x1_s, dts,
            As_1, Bs, Cs, Ds_1, z=None,
            delta_bias=dt_projs_bias_1,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert y_1.dtype == torch.float

        #########################################
        y_2 = self.selective_scan(
            x2_s, dts,
            As_2, Bs, Cs, Ds_2, z=None,
            delta_bias=dt_projs_bias_2,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert y_2.dtype == torch.float

        #########################################
        y_3 = self.selective_scan(
            x3_s, dts,
            As_3, Bs, Cs, Ds_3, z=None,
            delta_bias=dt_projs_bias_3,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert y_3.dtype == torch.float

        #########################################
        y_4 = self.selective_scan(
            x4_s, dts,
            As_4, Bs, Cs, Ds_4, z=None,
            delta_bias=dt_projs_bias_4,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert y_4.dtype == torch.float


        return torch.squeeze(y_1, dim=1), torch.squeeze(y_2, dim=1), torch.squeeze(y_3, dim=1), torch.squeeze(y_4, dim=1)

    def forward(self, x_1, x_2, x_3, x_4,  **kwargs):

        x_1_in, x_2_in, x_3_in, x_4_in = x_1, x_2, x_3, x_4

        Bs, Ls,  Cs = x_1.shape

        xz_1 = self.in_projs[0](x_1)
        xz_2 = self.in_projs[1](x_2)
        xz_3 = self.in_projs[2](x_3)
        xz_4 = self.in_projs[3](x_4)



        x_1, z_1 = xz_1.chunk(2, dim=-1)  # (b, l, c)
        x_2, z_2 = xz_2.chunk(2, dim=-1)  # (b, l, c)
        x_3, z_3 = xz_3.chunk(2, dim=-1)  # (b, l, c)
        x_4, z_4 = xz_4.chunk(2, dim=-1)  # (b, l, c)

        x_1, x_2, x_3, x_4 = x_1.permute(0, 2, 1), x_2.permute(0, 2, 1), x_3.permute(0, 2, 1), x_4.permute(0, 2, 1)         # (b, c, l)

        x_1, x_2, x_3, x_4 = self.act(self.conv1ds[0](x_1)), self.act(self.conv1ds[1](x_2)), self.act(self.conv1ds[2](x_3)), self.act(self.conv1ds[3](x_4))



        y_1, y_2, y_3, y_4 = self.forward_core(x_1, x_2, x_3, x_4)

        y_1, y_2, y_3, y_4 = y_1.permute(0, 2, 1), y_2.permute(0, 2, 1), y_3.permute(0, 2, 1), y_4.permute(0, 2, 1) # (b, l, c)
        y_1, y_2, y_3, y_4 = self.out_norm(y_1), self.out_norm(y_2), self.out_norm(y_3), self.out_norm(y_4)
        y_1, y_2, y_3, y_4 = y_1 * F.silu(z_1), y_2 * F.silu(z_2), y_3 * F.silu(z_3), y_4 * F.silu(z_4)
        out_y_1, out_y_2, out_y_3, out_y_4 = x_1_in + self.out_proj(y_1), x_2_in + self.out_proj(y_2), x_3_in + self.out_proj(y_3), x_4_in + self.out_proj(y_4)

        return out_y_1, out_y_2, out_y_3, out_y_4




if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x1 = torch.randn(size=(8, 200, 256)).to(device)
    x2 = torch.randn(size=(8, 200, 256)).to(device)
    x3 = torch.randn(size=(8, 200, 256)).to(device)
    x4 = torch.randn(size=(8, 200, 256)).to(device)



    ssm = MutualSSM(d_model=256).to(device)

    y_1, y_2, y_3, y_4 = ssm(x1, x2, x3, x4)

    t=1

    # x = torch.randn(size=(17, 16, 256)).to(device)
    # x_bcd = torch.randn(size=(17, 16, 256)).to(device)
    #
    # ss2d = SSM(d_model=256).to(device)
    #
    # y = ss2d(x, x_bcd)
    #
    # print(y.shape)
