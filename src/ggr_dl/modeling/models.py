# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear, global_mean_pool
from torch_geometric.utils import softmax


class HGTEncoder(nn.Module):
    """
    Heterogeneous encoder:
      x_dict -> z_dict
    PyG 的元信息，包含 (node_types, edge_types)
    """
    def __init__(
        self,
        metadata,
        in_dims: dict,        # e.g. {"gene":1024,"region":1024,"window":1024,"tf":1024}
        hid_dim: int = 256,
        out_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout

        self.lin_in = nn.ModuleDict({nt: Linear(in_dims[nt], hid_dim) for nt in in_dims})
        #self.convs = nn.ModuleList([HGTConv(hid_dim, hid_dim, metadata, heads=heads, group="sum")
        #                            for _ in range(layers)])
        self.convs = nn.ModuleList([HGTConv(hid_dim, hid_dim, metadata, heads=heads)
                                    for _ in range(layers)])
        
        self.lin_out = nn.ModuleDict({nt: Linear(hid_dim, out_dim) for nt in in_dims})

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            nt: F.dropout(F.relu(self.lin_in[nt](x)), p=self.dropout, training=self.training)
            for nt, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        z_dict = {k: self.lin_out[k](v) for k, v in x_dict.items()}
        return z_dict


class HGTRegressor(nn.Module):
    """
    Graph-level regressor for continuous trait on batched HeteroData.

    HGTEncoder
    -> pooling
    -> MLP
    -> 回归值

    readout_ntype:
      - "gene" (common): pool only gene nodes per graph
      - or you can later extend to concat multiple types
    """
    def __init__(
        self,
        metadata,
        in_dims: dict,
        hid_dim: int = 256,
        out_dim: int = 256,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.2,
        readout_ntype: str = "gene",
        readout: str = "mean",          # "mean" or "attention"
    ):
        super().__init__()
        assert readout in ("mean", "attention")
        self.readout_ntype = readout_ntype
        self.readout = readout
        self.dropout = dropout

        self.encoder = HGTEncoder(
            metadata=metadata,
            in_dims=in_dims,
            hid_dim=hid_dim,
            out_dim=out_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )

        if readout == "attention":
            self.att = nn.Linear(out_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def pool_batched(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # z: [N, D], batch: [N] -> [B, D]
        if self.readout == "mean":
            return global_mean_pool(z, batch)
        # attention pooling per-graph (segment softmax)
        w = self.att(z).squeeze(-1)             # [N]
        w = softmax(w, batch)                  # normalize within each graph
        g = global_mean_pool(z * w.unsqueeze(-1), batch) * (batch.bincount().clamp_min(1).float().unsqueeze(-1))
        #  mean_pool + 乘回节点数 ≈ sum pooling（更贴近 attention 加权和）
        return g

    def forward_heterodata(self, data):
        z_dict = self.encoder(data.x_dict, data.edge_index_dict)
        z = z_dict[self.readout_ntype]

        store = data[self.readout_ntype]
        if hasattr(store, "batch"):
            batch = store.batch
        else:
            # single graph: all nodes belong to graph 0
            batch = z.new_zeros(z.size(0), dtype=torch.long)

        g = self.pool_batched(z, batch)
        y_hat = self.mlp(g).squeeze(-1)  # [B]
        return y_hat, z_dict

    def forward(self, x_dict, edge_index_dict):
        # 保持向后兼容（旧接口），但新训练建议用 forward_heterodata
        z_dict = self.encoder(x_dict, edge_index_dict)
        z = z_dict[self.readout_ntype]
        g = z.mean(dim=0)
        return self.mlp(g).squeeze(-1)


def save_checkpoint(path: str, model: nn.Module, hparams: dict, extra: dict | None = None):
    payload = {"model": model.state_dict(), "hparams": hparams}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_from_checkpoint(ckpt_path: str, metadata, in_dims: dict, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    h = ckpt.get("hparams", {})
    model = HGTRegressor(
        metadata=metadata,
        in_dims=in_dims,
        hid_dim=h.get("hid_dim", 256),
        out_dim=h.get("out_dim", 256),
        heads=h.get("heads", 4),
        layers=h.get("layers", 2),
        dropout=h.get("dropout", 0.2),
        readout_ntype=h.get("readout_ntype", "gene"),
        readout=h.get("readout", "mean"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, ckpt


'''
HGTEncoder
    └── 输入投影
    └── 多层 HGTConv
    └── 输出投影

HGTRegressor
    └── Encoder
    └── Pooling (mean / attention)
    └── MLP
    └── 输出标量

'''