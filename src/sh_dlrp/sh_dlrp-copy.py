import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from src.sh_dlrp.mutual_ssm import MutualSSM
from src.sh_dlrp.att_pool import AttPoolBlock

from src.sh_dlrp.read_out import ModalAtt, ModalAttention


class SHDLRP(nn.Module):
    def __init__(self, extractor, extractor_choose, return_wsi_heatmap, feature_dim, top_k):
        super(SHDLRP, self).__init__()

        self.extractor = extractor

        self.top_k = top_k

        self.extractors = _get_clones(extractor, 3)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 7, 32),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.gate = attnet()

        # self.layer_norm = nn.LayerNorm(1000)

        # self.liners = nn.ModuleList([nn.Linear(1935, 4000) for _ in range(3)])

        self.liners = nn.ModuleList([nn.Linear(645, 8) for _ in range(3)])

        self.bus_linear = nn.Linear(49, 16)
        self.swt_linear = nn.Linear(58, 16)
        self.swv_linear = nn.Linear(52, 16)


        self.projs = nn.ModuleList([nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16)) for _ in range(7)])

        self.MutualSSM = MutualSSM(d_model=256)

        # self.pool = nn.Conv1d(
        #     in_channels=200,
        #     out_channels=1,
        #     kernel_size=1,
        #     padding=0,
        # )

        self.patch_pools = nn.ModuleList([AttPoolBlock(256) for _ in range(7)])
        # self.patch_pool = AttPoolBlock(256)

        self.report_proj = nn.Sequential(
            nn.LayerNorm(19),
            nn.Linear(19, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # self.dim_transoforms = nn.ModuleList([nn.Linear(768, 128) for _ in range(3)])

        self.extractor_choose = extractor_choose

        self.modal_att = ModalAttention(embed_dim=16, modal_num=7)

        self.wsi_proj = nn.Linear(1000, 16)

        self.return_wsi_heatmap = return_wsi_heatmap



    def forward(self, x):

        swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, sample_name = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]



        swt_emb = self.extractors[0](swt)
        swv_emb = self.extractors[1](swv)
        bus_emb = self.extractors[2](bus)

        # elif self.extractor_choose == "vit":
        #     swt_emb = self.dim_transoforms[0](self.extractors[0](swt))
        #     swv_emb = self.dim_transoforms[1](self.extractors[1](swv))
        #     bus_emb = self.dim_transoforms[2](self.extractors[2](bus))

        # else:
        #     assert 0, "Choose a extractor!"

        # swt_fea = self.liners[0](swt_fea).view(swt_fea.shape[0], top_k_num - swt_emb.shape[1], swt_emb.shape[2])
        # swv_fea = self.liners[1](swv_fea).view(swv_fea.shape[0], top_k_num - swv_emb.shape[1], swv_emb.shape[2])
        # bus_fea = self.liners[2](bus_fea).view(bus_fea.shape[0], top_k_num - bus_emb.shape[1], bus_emb.shape[2])

        # swt_fea = self.liners[0](swt_fea)
        # swv_fea = self.liners[1](swv_fea)
        # bus_fea = self.liners[2](bus_fea)

        swt_fea = self.swt_linear(swt_fea)
        swv_fea = self.swv_linear(swv_fea)
        bus_fea = self.bus_linear(bus_fea)

        # swt_comb = torch.cat((swt_emb, swt_fea), dim=1)
        # swv_comb = torch.cat((swv_emb, swv_fea), dim=1)
        # bus_comb = torch.cat((bus_emb, bus_fea), dim=1)

        wsi_masks = []

        for batch in range(wsi.shape[0]):
            mask_idx = mask[batch]
            wsi_batch = wsi[batch]
            mask_wsi = wsi_batch[mask_idx]
            wsi_masks.append(mask_wsi)

        wsi_heatmaps = []
        for batch, mask_wsi in enumerate(wsi_masks):

            A = self.gate(mask_wsi)

            A = A.permute(1, 0)
            A = F.softmax(A, dim=-1)

            if self.return_wsi_heatmap:
                wsi_heatmaps.append({"sample": sample_name[batch], "heatmap": A})

            batch_top_p_ids = torch.topk(A, self.top_k)[1][-1]
            # batch_top_p_ids = topk.values[batch][0]

            batch_top_p = torch.index_select(mask_wsi, dim=0, index=batch_top_p_ids)

            if batch == 0:
                wsi_comb = torch.unsqueeze(batch_top_p, dim=0)
            else:
                wsi_comb = torch.cat((wsi_comb, torch.unsqueeze(batch_top_p, dim=0)), dim=0)

        # for batch, mask_wsi in enumerate(wsi_masks):
        #
        #     one_batch = torch.sum(mask_wsi, dim=0) / mask_wsi.shape[0]
        #
        #     if batch == 0:
        #         wsi_comb = torch.unsqueeze(one_batch, dim=0)
        #     else:
        #         wsi_comb = torch.cat((wsi_comb, torch.unsqueeze(one_batch, dim=0)), dim=0)

        # wsi_comb, swt_comb, swv_comb, bus_comb = self.MutualSSM(self.projs[0](wsi_comb), self.projs[1](swt_comb), self.projs[2](swv_comb), self.projs[3](bus_comb))

        wsi_comb = self.wsi_proj(wsi_comb)

        wsi_comb, swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea = self.projs[0](wsi_comb), self.projs[1](swt_emb), self.projs[2](swv_emb), self.projs[3](bus_emb), self.projs[4](swt_fea), self.projs[5](swv_fea), self.projs[6](bus_fea)

        wsi_comb = self.patch_pools[0](wsi_comb)

        # if not self.extractor_choose == "resnet":
        #     swt_emb = self.patch_pools[1](swt_emb)
        #     swv_emb = self.patch_pools[2](swv_emb)
        #     bus_emb = self.patch_pools[3](bus_emb)

        # swt_fea = self.patch_pools[4](swt_fea)
        # swv_fea = self.patch_pools[5](swv_fea)
        # bus_fea = self.patch_pools[6](bus_fea)

        # top_p = torch.squeeze(self.pool(top_p), dim=1)
        # swt_comb = torch.squeeze(self.pool(swt_comb), dim=1)
        # swv_comb = torch.squeeze(self.pool(swv_comb), dim=1)
        # bus_comb = torch.squeeze(self.pool(bus_comb), dim=1)

        # top_p = torch.mean(top_p, dim=1)
        #
        # swt_comb = torch.mean(swt_comb, dim=1)
        # swv_comb = torch.mean(swv_comb, dim=1)
        # bus_comb = torch.mean(bus_comb, dim=1)

        rep_fea = self.report_proj(report)

        # wsi_comb, swt_comb, swv_comb, bus_comb = self.modal_att(wsi_comb, swt_comb, swv_comb, bus_comb)

        # out = torch.cat((swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea, wsi_comb), dim=-1)
        # out = torch.cat((swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea), dim=-1)
        # out = wsi_comb

        out = self.modal_att((swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea, wsi_comb))

        # top_p = self.layer_norm(torch.sum(top_p, dim=1))
        if self.return_wsi_heatmap:
            return self.classifier(out), wsi_heatmaps
        return self.classifier(out)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A

class attnet(nn.Module):
    def __init__(self):
        super(attnet, self).__init__()

        attention_net = Attn_Net_Gated(512, D=256, dropout=True, n_classes=1)
        fc = [nn.Linear(1000, 512), nn.ReLU(), nn.Dropout(0.25)]
        fc.append(attention_net)

        self.attn = nn.Sequential(*fc)


    def forward(self, x):
        return self.attn(x)
