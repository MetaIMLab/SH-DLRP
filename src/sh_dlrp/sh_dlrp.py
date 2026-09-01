import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, knn_graph


class GATModel(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)


class SHDLRP(nn.Module):
    def __init__(
        self,
        extractor,
        extractor_choose,
        return_wsi_heatmap,
        modal,
        feature_dim,
        top_k,
        train_patch_root="./dataset/sysucc-patch",
        test_patch_root="./dataset/sysush-patch",
    ):
        super().__init__()
        self.extractor = extractor
        self.extractor_choose = extractor_choose
        self.top_k = int(top_k)
        self.modal = modal
        self.return_wsi_heatmap = return_wsi_heatmap
        self.train_patch_root = train_patch_root
        self.test_patch_root = test_patch_root

        self.extractors = _get_clones(extractor, 3)
        self.bus_gat = GATModel(768, 16)
        self.swt_gat = GATModel(768, 16)
        self.swv_gat = GATModel(768, 16)
        self.wsi_gat = GATModel(1000, 16)
        self.us_projs = nn.ModuleList([nn.LazyLinear(16) for _ in range(3)])

        self.bus_linear = nn.Linear(1935, 16)
        self.swt_linear = nn.Linear(1935, 16)
        self.swv_linear = nn.Linear(1935, 16)
        self.projs = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16)) for _ in range(7)]
        )
        self.gate = attnet()

        classifier_input = {"all": 16 * 7, "us": 16 * 6, "rad": 16 * 3, "wsi": 16}
        if modal not in classifier_input:
            raise ValueError("Unsupported modal '{}'".format(modal))
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input[modal], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    @staticmethod
    def _knn_edges(coords, k=4):
        if coords.shape[0] < 2:
            return torch.empty((2, 0), dtype=torch.long, device=coords.device)
        return knn_graph(coords, k=min(k, coords.shape[0] - 1), loop=False)

    def _encode_ultrasound(self, images):
        embeddings = [
            extractor(image)
            for extractor, image in zip(self.extractors, images)
        ]

        if getattr(self.extractors[0], "intermediate_feature", False):
            token_count = embeddings[0].shape[1]
            grid_size = int(token_count ** 0.5)
            if grid_size * grid_size != token_count:
                raise ValueError("ViT patch token count must form a square grid")

            rows = torch.arange(grid_size, dtype=torch.float32, device=embeddings[0].device)
            cols = torch.arange(grid_size, dtype=torch.float32, device=embeddings[0].device)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
            coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
            edge_index = self._knn_edges(coords)
            graphs = (self.swt_gat, self.swv_gat, self.bus_gat)

            pooled = []
            for embedding, graph in zip(embeddings, graphs):
                samples = []
                for sample in embedding:
                    node_features = graph(sample, edge_index.to(sample.device))
                    samples.append(node_features.mean(dim=0))
                pooled.append(torch.stack(samples, dim=0))
            return tuple(pooled)

        outputs = []
        for index, embedding in enumerate(embeddings):
            if embedding.ndim == 4:
                embedding = F.adaptive_avg_pool2d(embedding, (1, 1)).flatten(1)
            if embedding.ndim != 2:
                raise ValueError(
                    "Extractor output must be [batch, dim] or [batch, tokens, dim]"
                )
            outputs.append(self.us_projs[index](embedding))
        return tuple(outputs)

    def _encode_wsi(self, wsi, mask, sample_name, is_test):
        patch_root = self.test_patch_root if is_test else self.train_patch_root
        wsi_features = []
        wsi_heatmaps = []

        for batch, name in enumerate(sample_name):
            mask_wsi = wsi[batch][mask[batch]]
            coord_dir = os.path.join(patch_root, name)
            if not os.path.isdir(coord_dir):
                raise FileNotFoundError("Missing WSI patch directory '{}'".format(coord_dir))

            coord_files = sorted(
                file_name
                for file_name in os.listdir(coord_dir)
                if file_name.lower().endswith(".png")
            )
            if len(coord_files) != mask_wsi.shape[0]:
                raise ValueError(
                    "WSI feature/coordinate count mismatch for '{}': {} vs {}".format(
                        name, mask_wsi.shape[0], len(coord_files)
                    )
                )

            sample_coords = []
            for coord_file in coord_files:
                coord = coord_file[:-4].split("_")[-1].split("-")[-4:]
                if len(coord) != 4:
                    raise ValueError("Cannot parse WSI coordinates from '{}'".format(coord_file))
                x1, y1, x2, y2 = (int(value) / 32 for value in coord)
                sample_coords.append(((x1 + x2) / 2, (y1 + y2) / 2))
            sample_coords = torch.tensor(
                sample_coords, dtype=torch.float32, device=wsi.device
            )

            attention = F.softmax(self.gate(mask_wsi).transpose(0, 1), dim=-1)
            if self.return_wsi_heatmap:
                wsi_heatmaps.append(
                    {"sample": name, "heatmap": attention.detach()}
                )

            top_count = min(self.top_k, mask_wsi.shape[0])
            if top_count == 0:
                raise ValueError("WSI sample '{}' has no valid tiles".format(name))
            top_ids = torch.topk(attention[0], top_count).indices
            top_weights = attention[0].index_select(0, top_ids)
            top_weights = top_weights / top_weights.sum().clamp_min(
                torch.finfo(top_weights.dtype).eps
            )

            selected_features = mask_wsi.index_select(0, top_ids)
            selected_coords = sample_coords.index_select(0, top_ids)
            selected_features = self.wsi_gat(
                selected_features,
                self._knn_edges(selected_coords).to(selected_features.device),
            )
            selected_features = self.projs[0](selected_features)
            wsi_features.append(
                (selected_features * top_weights.unsqueeze(-1)).sum(dim=0)
            )

        return torch.stack(wsi_features, dim=0), wsi_heatmaps

    def forward(self, x, is_test=False, tsne_visual=False, embed=False):
        swt, swv, bus, swt_fea, swv_fea, bus_fea, wsi, mask, report, sample_name = x
        use_ultrasound = self.modal in {"all", "us"}
        use_wsi = self.modal in {"all", "wsi"}

        if use_ultrasound:
            swt_emb, swv_emb, bus_emb = self._encode_ultrasound((swt, swv, bus))
            swt_emb, swv_emb, bus_emb = (
                self.projs[1](swt_emb),
                self.projs[2](swv_emb),
                self.projs[3](bus_emb),
            )

        swt_fea = self.projs[4](self.swt_linear(swt_fea))
        swv_fea = self.projs[5](self.swv_linear(swv_fea))
        bus_fea = self.projs[6](self.bus_linear(bus_fea))

        if use_wsi:
            wsi_fea, wsi_heatmaps = self._encode_wsi(
                wsi, mask, sample_name, is_test
            )
        else:
            wsi_heatmaps = []

        if self.modal == "all":
            out = torch.cat(
                (swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea, wsi_fea),
                dim=-1,
            )
        elif self.modal == "us":
            out = torch.cat(
                (swt_emb, swv_emb, bus_emb, swt_fea, swv_fea, bus_fea),
                dim=-1,
            )
        elif self.modal == "rad":
            out = torch.cat((swt_fea, swv_fea, bus_fea), dim=-1)
        else:
            out = wsi_fea

        if tsne_visual:
            return out
        if embed:
            return self.classifier(out), out
        if self.return_wsi_heatmap:
            return self.classifier(out), wsi_heatmaps
        return self.classifier(out)


def _get_clones(module, count):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class Attn_Net_Gated(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, dropout=False, n_classes=1):
        super().__init__()
        attention_a = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        attention_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            attention_a.append(nn.Dropout(0.25))
            attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.attention_c(self.attention_a(x) * self.attention_b(x))


class attnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            Attn_Net_Gated(512, hidden_dim=256, dropout=True, n_classes=1),
        )

    def forward(self, x):
        return self.attn(x)
