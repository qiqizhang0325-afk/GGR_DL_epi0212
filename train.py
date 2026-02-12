#!/usr/bin/env python3
# train.py (mutant_graph version)
# train.py (mutant_graph version)
#
# 功能：
#   1) 从 graphs_dir sample graph（.pt files，including PyG HeteroData）
#   2) 从 labels_tsv 读取每个 sample_id 的回归标签
#   3) 构建 Dataset / DataLoader，训练 HGTRegressor 做图级回归
#   4) 记录训练日志、保存最优 checkpoint、导出验证集预测结果

import os
import re
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from models import HGTRegressor, save_checkpoint

# 用于从文件名中提取 sample_id：
# sample_<sample_id>_mutant_graph.pt
PT_RE = re.compile(r"^sample_(.+)_mutant_graph\.pt$")


def discover_graphs(graphs_dir: str):
    '''
    扫描 graphs_dir，发现所有符合命名规范的图文件，返回 sample_id -> graph_path 的映射。
    规范的文件名示例：sample_123_mutant_graph.pt -> sample_id=123
    sid2pt : dict[str, str]
    {'9367_9367': '/lustre/BIF/nobackup/zhang479/new_PlantCaduceus/PlantCaduceus/arb_chr2_test0211/result/sample_graphs/9367_9367/sample_9367_9367_mutant_graph.pt'}
    '''
    sid2pt = {}
    for p in Path(graphs_dir).rglob("sample_*_mutant_graph.pt"):
        m = PT_RE.match(p.name)
        if m:
            sid2pt[m.group(1)] = str(p)
    return sid2pt


def read_labels_tsv(path: str):
    labels = {}
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert "sample_id" in header and "label" in header
        si = header.index("sample_id")
        yi = header.index("label")
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            labels[cols[si]] = float(cols[yi])
    return labels


class GraphLabelDataset(torch.utils.data.Dataset):
    '''Dataset for graph-level regression. Each item is a (graph, label) pair.
    graph is loaded from .pt file, label is read from labels dict.
    Only samples with both graph and label are included.
    sid2pt: dict[sample_id, graph_path]
    labels: dict[sample_id, label_value]
    
    input:
    sid2pt = {  '9367_9367': '/lustre/BIF/nobackup/zhang479/new_PlantCaduceus/PlantCaduceus/arb_chr2_test0211/result/sample_graphs/9367_9367/sample_9367_9367_mutant_graph.pt'}
    labels = {'9367_9367': 0.5, 'other_sample': 1.2, ...}
    output: dataset with items like:
    [       ( '9367_9367', '/lustre/BIF/nobackup/zhang479/new_PlantCaduceus/PlantCaduceus/arb_chr2_test0211/result/sample_graphs/9367_9367/sample_9367_9367_mutant_graph.pt', 0.5 ),
            ( 'other_sample', '/path/to/other_sample_mutant_graph.pt', 1.2 ),
            ...  ]         
    '''
    def __init__(self, sid2pt: dict, labels: dict):
        self.items = []
        for sid, pt in sid2pt.items():
            if sid in labels:
                self.items.append((sid, pt, labels[sid]))
        if not self.items:
            raise ValueError("No matched (graph, label) pairs. Check graphs_dir and labels.tsv")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, pt, y = self.items[idx]
        pack = torch.load(pt, map_location="cpu")
        data = pack["data"]
        data.y = torch.tensor([y], dtype=torch.float)  # [1]
        data.sid = sid
        return data


def pearsonr_torch(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)


def main():
    '''HGTRegressor:
    HGTEncoder -> pooling -> MLP -> 回归值
    pooling:
        - mean: 直接 mean_pool
        - attention: 先算 attention 权重（节点级别），然后加权求和（更贴近 sum pooling）
    
    训练流程：
    1) 从 graphs_dir 发现图文件，构建 sid2pt
    2) 从 labels_tsv 读取标签，构建 labels dict
    3) 构建 GraphLabelDataset，自动过滤掉没有图或标签的 sample
    4) 划分训练/验证集，构建 DataLoader
    5) 定义模型、优化器、损失函数
    6) 训练循环：每 epoch 训练 + 验证，记录日志，保存最优 checkpoint
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--labels_tsv", required=True)
    ap.add_argument("--outdir", required=True)

    # training hyperparameters
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--train_ratio", type=float, default=0.8)

    # model hyperparameters
    ap.add_argument("--hid_dim", type=int, default=256)
    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--readout_ntype", default="gene")
    ap.add_argument("--readout", default="mean")  # mean or attention

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) 从 graphs_dir 发现图文件，构建 sid2pt
    # 2) 从 labels_tsv 读取标签，构建 labels dict
    # 3) 构建 GraphLabelDataset，自动过滤掉没有图或标签的 sample
    sid2pt = discover_graphs(args.graphs_dir)
    print("Found graphs:", len(sid2pt), list(sid2pt.keys())[:5])
    labels = read_labels_tsv(args.labels_tsv)
    print("Found labels:", len(labels))
    dataset = GraphLabelDataset(sid2pt, labels)

    # split
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(n * args.train_ratio)
    train_idx, val_idx = idx[:split], idx[split:]

    # save split lists
    with open(os.path.join(args.outdir, "train_samples.txt"), "w") as f:
        for i in train_idx:
            f.write(dataset.items[i][0] + "\n")
    with open(os.path.join(args.outdir, "val_samples.txt"), "w") as f:
        for i in val_idx:
            f.write(dataset.items[i][0] + "\n")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # infer metadata/in_dims from one graph
    # ===== 从一个样本推断 metadata 与每类节点输入维度 in_dims =====
    sample0 = dataset[0]
    metadata = sample0.metadata()
    in_dims = {nt: sample0[nt].x.size(1) for nt in sample0.node_types}

    print("Metadata:", metadata)
    print("in_dims:", in_dims)


    model = HGTRegressor(
        metadata=metadata,
        in_dims=in_dims,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        readout_ntype=args.readout_ntype,
        readout=args.readout,
    ).to(args.device)

    # 优化器和损失函数
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # MSELoss 用于回归任务
    loss_fn = nn.MSELoss()

    log_path = os.path.join(args.outdir, "training_log.tsv")
    with open(log_path, "w") as f:
        f.write("epoch\ttrain_mse\tval_mse\tval_r\tval_r2\n")

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {ep} [train]"):
            batch = batch.to(args.device)
            y_true = batch.y.view(-1).to(args.device)     # [B]
            y_pred, _ = model.forward_heterodata(batch)   # [B]
            loss = loss_fn(y_pred, y_true)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            bs = y_true.size(0)
            total += loss.item() * bs
            n_train += bs

        train_mse = total / max(1, n_train)

        model.eval()
        with torch.no_grad():
            preds, ys = [], []
            for batch in val_loader:
                batch = batch.to(args.device)
                y_true = batch.y.view(-1).to(args.device)
                y_pred, _ = model.forward_heterodata(batch)
                preds.append(y_pred)
                ys.append(y_true)
            preds = torch.cat(preds) if preds else torch.empty(0, device=args.device)
            ys = torch.cat(ys) if ys else torch.empty(0, device=args.device)

            val_mse = F.mse_loss(preds, ys).item() if ys.numel() else float("nan")
            r = float(pearsonr_torch(preds, ys).item()) if ys.numel() else float("nan")
            r2 = 1.0 - val_mse / (ys.var().item() + 1e-8) if ys.numel() else float("nan")

        with open(log_path, "a") as f:
            f.write(f"{ep}\t{train_mse:.6f}\t{val_mse:.6f}\t{r:.4f}\t{r2:.4f}\n")

        print(f"[Epoch {ep}] train MSE={train_mse:.4f} | val MSE={val_mse:.4f} | r={r:.3f} | R2={r2:.3f}")

        if val_mse < best_val:
            best_val = val_mse
            hparams = {
                "hid_dim": args.hid_dim,
                "out_dim": args.out_dim,
                "heads": args.heads,
                "layers": args.layers,
                "dropout": args.dropout,
                "readout_ntype": args.readout_ntype,
                "readout": args.readout,
            }
            save_checkpoint(
                os.path.join(args.outdir, "checkpoint.pt"),
                model=model,
                hparams=hparams,
                extra={"epoch": ep, "val_mse": val_mse, "in_dims": in_dims},
            )

    # save val predictions
    pred_path = os.path.join(args.outdir, "predictions_val.tsv")
    model.eval()
    with torch.no_grad():
        with open(pred_path, "w") as f:
            f.write("sample_id\ty_true\ty_pred\n")
            for batch in val_loader:
                batch = batch.to(args.device)
                y_true = batch.y.view(-1)
                y_pred, _ = model.forward_heterodata(batch)
                # sid is a python object list in batch; PyG keeps it in batch.sid as list-like
                sids = batch.sid
                for sid, yt, yp in zip(sids, y_true.tolist(), y_pred.tolist()):
                    f.write(f"{sid}\t{yt:.6f}\t{yp:.6f}\n")

    print(f"✅ saved: {pred_path}")
    print("✅ done.")


if __name__ == "__main__":
    main()
