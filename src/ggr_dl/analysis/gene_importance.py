#!/usr/bin/env python3
# 20_gene_importance.py (mutant_graph version, fixed for .batch)

import os
import re
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch

from ggr_dl.modeling.models import load_model_from_checkpoint

PT_RE = re.compile(r"^sample_(.+)_mutant_graph\.pt$")


def read_sample_list(path: str):
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def discover_graphs(graphs_dir: str):
    """Recursively scan graphs_dir and build sid -> full path."""
    graphs_dir = Path(graphs_dir)
    sid2pt = {}
    for p in graphs_dir.rglob("sample_*_mutant_graph.pt"):
        m = PT_RE.match(p.name)
        if m:
            sid2pt[m.group(1)] = str(p)
    return sid2pt


def ensure_batched_heterodata(data, readout_ntype: str):
    """
    Ensure data[readout_ntype].batch exists.
    For a single HeteroData graph, wrap into a Batch of size 1.
    """
    store = data[readout_ntype]
    if hasattr(store, "batch"):
        return data
    return Batch.from_data_list([data])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_pt", required=True, help="reference graph pack (.pt) containing keys: data, meta")
    ap.add_argument("--checkpoint_pt", required=True, help="model checkpoint .pt")
    ap.add_argument("--graphs_dir", required=True, help="directory containing sample graphs (recursive)")
    ap.add_argument("--sample_list", required=True, help="val_samples.txt (sample_id per line)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- sanity checks ----
    if not os.path.exists(args.reference_pt):
        raise FileNotFoundError(f"[reference_pt not found] {args.reference_pt}")
    if not os.path.exists(args.checkpoint_pt):
        raise FileNotFoundError(f"[checkpoint_pt not found] {args.checkpoint_pt}")
    if not os.path.exists(args.sample_list):
        raise FileNotFoundError(f"[sample_list not found] {args.sample_list}")
    if not os.path.isdir(args.graphs_dir):
        raise NotADirectoryError(f"[graphs_dir not a directory] {args.graphs_dir}")

    samples = read_sample_list(args.sample_list)
    if args.max_samples != -1:
        samples = samples[: args.max_samples]
    if len(samples) == 0:
        raise RuntimeError(f"[sample_list is empty] {args.sample_list}")

    # ---- load reference ----
    ref = torch.load(args.reference_pt, map_location="cpu")
    if "data" not in ref or "meta" not in ref:
        raise KeyError(f"[reference_pt format error] expect keys 'data' and 'meta', got keys={list(ref.keys())}")

    ref_data = ref["data"].to(args.device)
    meta = ref["meta"]
    if "gene_ids" not in meta:
        raise KeyError(f"[reference_pt meta error] expect meta['gene_ids'], got meta keys={list(meta.keys())}")
    gene_ids = meta["gene_ids"]

    # in_dims must match model
    in_dims = {nt: ref_data[nt].x.size(1) for nt in ref_data.node_types}

    # ---- load model ----
    model, _ = load_model_from_checkpoint(
        args.checkpoint_pt,
        metadata=ref_data.metadata(),
        in_dims=in_dims,
        device=args.device
    )
    model.eval()

    readout_ntype = getattr(model, "readout_ntype", "gene")

    # ---- index graphs once ----
    sid2pt = discover_graphs(args.graphs_dir)
    if len(sid2pt) == 0:
        raise RuntimeError(f"[no graphs found] under {args.graphs_dir} with pattern sample_*_mutant_graph.pt")

    # ---- accumulate scores (still output gene_ids order) ----
    scores = torch.zeros((len(gene_ids),), device=args.device)
    used = 0
    missing = 0

    for sid in tqdm(samples, desc="Gradient×Input (val only)"):
        pt = sid2pt.get(sid)
        if pt is None:
            missing += 1
            continue

        pack = torch.load(pt, map_location="cpu")
        data = pack["data"].to(args.device)

        # ensure .batch exists for model readout ntype
        data = ensure_batched_heterodata(data, readout_ntype)

        y_pred, z_dict = model.forward_heterodata(data)

        # IMPORTANT: if you only care about gene attribution, keep "gene" here.
        # If you want it to follow model readout type, use z_dict[readout_ntype].
        z = z_dict["gene"]
        z.retain_grad()

        model.zero_grad(set_to_none=True)
        y_pred.sum().backward()

        if z.grad is None:
            raise RuntimeError(f"[grad is None] sid={sid} (check autograd path)")

        sc = (z * z.grad).abs().sum(dim=1).detach()
        scores += sc
        used += 1

    if used == 0:
        raise RuntimeError(
            f"[no samples used] sample_list={args.sample_list}. "
            f"Graphs missing for all samples? (missing={missing}/{len(samples)})"
        )

    scores = (scores / used).detach().cpu().numpy()
    order = np.argsort(-scores)

    out_path = f"{args.out_prefix}.gene_importance.tsv"
    with open(out_path, "w") as f:
        f.write("id\tscore\trank\n")
        for rank, idx in enumerate(order, start=1):
            f.write(f"{gene_ids[idx]}\t{scores[idx]:.6e}\t{rank}\n")

    print(f"✅ saved: {out_path} (n_used={used}, missing_graph={missing})")


if __name__ == "__main__":
    main()
