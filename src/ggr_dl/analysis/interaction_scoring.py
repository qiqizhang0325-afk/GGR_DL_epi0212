#!/usr/bin/env python3
# 30_interaction_scoring.py (mutant_graph version, fixed for subdir graphs + fast masking)
# UPDATED: decode tf node indices to real TF ids using meta["tf_ids"]

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


def read_top_ids(tsv_path: str, topk: int):
    ids = []
    with open(tsv_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            ids.append(line.split("\t")[0])
            if len(ids) >= topk:
                break
    return set(ids)


def edge_score(z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
    # z_src,z_dst: [E,D] -> [E]
    return (z_src * z_dst).abs().sum(dim=1)


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
    Ensure data[readout_ntype].batch exists (batch_size=1).
    Note: models.py often handles missing batch; keeping for consistency.
    """
    store = data[readout_ntype]
    if hasattr(store, "batch"):
        return data
    return Batch.from_data_list([data])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_pt", required=True)
    ap.add_argument("--checkpoint_pt", required=True)
    ap.add_argument("--graphs_dir", required=True)

    ap.add_argument("--gene_importance_tsv", required=True)
    ap.add_argument("--top_genes", type=int, default=2000)

    ap.add_argument("--sample_list", required=True, help="val_samples.txt")
    ap.add_argument("--max_samples", type=int, default=-1)

    ap.add_argument("--top_edges", type=int, default=10000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- sanity checks ----
    if not os.path.exists(args.reference_pt):
        raise FileNotFoundError(f"[reference_pt not found] {args.reference_pt}")
    if not os.path.exists(args.checkpoint_pt):
        raise FileNotFoundError(f"[checkpoint_pt not found] {args.checkpoint_pt}")
    if not os.path.isdir(args.graphs_dir):
        raise NotADirectoryError(f"[graphs_dir not a directory] {args.graphs_dir}")
    if not os.path.exists(args.gene_importance_tsv):
        raise FileNotFoundError(f"[gene_importance_tsv not found] {args.gene_importance_tsv}")
    if not os.path.exists(args.sample_list):
        raise FileNotFoundError(f"[sample_list not found] {args.sample_list}")

    # ---- load reference ----
    ref = torch.load(args.reference_pt, map_location="cpu")
    ref_data = ref["data"].to(args.device)
    meta = ref["meta"]

    gene_ids = meta["gene_ids"]
    region_ids = meta.get("region_ids", [])
    window_ids = meta.get("window_ids", [])
    tf_ids = meta.get("tf_ids", [])  # <-- NEW: decode tf node indices

    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}
    top_gene_set = read_top_ids(args.gene_importance_tsv, args.top_genes)

    # vectorized gene_keep mask
    gene_keep = torch.zeros(len(gene_ids), dtype=torch.bool, device=args.device)
    for g in top_gene_set:
        gi = gene_id_to_idx.get(g, None)
        if gi is not None:
            gene_keep[gi] = True

    # ---- load model ----
    in_dims = {nt: ref_data[nt].x.size(1) for nt in ref_data.node_types}
    model, _ = load_model_from_checkpoint(
        args.checkpoint_pt,
        metadata=ref_data.metadata(),
        in_dims=in_dims,
        device=args.device
    )
    model.eval()
    readout_ntype = getattr(model, "readout_ntype", "gene")

    edge_types = list(ref_data.edge_types)

    # ---- precompute edge masks on reference topology (fast) ----
    edge_masks = {}
    for et in edge_types:
        src_type, rel, dst_type = et
        ei = ref_data[et].edge_index.to(args.device)
        src, dst = ei[0], ei[1]

        # Keep edges if either endpoint is a top gene (only when gene is involved)
        if src_type == "gene" or dst_type == "gene":
            mask = torch.zeros(src.size(0), dtype=torch.bool, device=args.device)
            if src_type == "gene":
                mask |= gene_keep[src]
            if dst_type == "gene":
                mask |= gene_keep[dst]
            edge_masks[et] = mask
        else:
            edge_masks[et] = None

    # ---- discover graph paths once (subdir aware) ----
    sid2pt = discover_graphs(args.graphs_dir)
    if len(sid2pt) == 0:
        raise RuntimeError(f"[no graphs found] under {args.graphs_dir} with pattern sample_*_mutant_graph.pt")

    # ---- sample list ----
    samples = read_sample_list(args.sample_list)
    if args.max_samples != -1:
        samples = samples[:args.max_samples]
    if len(samples) == 0:
        raise RuntimeError(f"[sample_list is empty] {args.sample_list}")

    accum = {}   # et -> sum(scores per edge after masking)  (tensor shape [E_masked])
    counts = {}  # et -> number of graphs accumulated (scalar)

    used = 0
    missing = 0

    for sid in tqdm(samples, desc="Averaging interaction scores (val graphs)"):
        pt = sid2pt.get(sid)
        if pt is None:
            missing += 1
            continue

        pack = torch.load(pt, map_location="cpu")
        data = pack["data"].to(args.device)
        data = ensure_batched_heterodata(data, readout_ntype)

        with torch.no_grad():
            _, z_dict = model.forward_heterodata(data)

        for et in edge_types:
            src_type, rel, dst_type = et

            # IMPORTANT:
            # We assume edge topology is consistent with reference_pt (common in your pipeline).
            ei = ref_data[et].edge_index.to(args.device)
            src, dst = ei[0], ei[1]

            m = edge_masks[et]
            if m is not None:
                src = src[m]
                dst = dst[m]
                if src.numel() == 0:
                    continue

            sc = edge_score(z_dict[src_type][src], z_dict[dst_type][dst]).detach()

            if et not in accum:
                accum[et] = sc.clone()
                counts[et] = 1
            else:
                accum[et] += sc
                counts[et] += 1

        used += 1

    def id_of(ntype, idx):
        """Decode node index -> human-readable id string."""
        idx = int(idx)

        if ntype == "gene":
            if 0 <= idx < len(gene_ids):
                return gene_ids[idx]
            return f"GENE_IDX_OUT_OF_RANGE:{idx}"

        if ntype == "tf":
            if tf_ids and 0 <= idx < len(tf_ids):
                return tf_ids[idx]
            return f"TF_IDX_OUT_OF_RANGE:{idx}"

        if ntype == "region":
            if region_ids and 0 <= idx < len(region_ids):
                return region_ids[idx]
            return f"REGION_IDX_OUT_OF_RANGE:{idx}"

        if ntype == "window":
            if window_ids and 0 <= idx < len(window_ids):
                return window_ids[idx]
            return f"WINDOW_IDX_OUT_OF_RANGE:{idx}"

        # fallback
        return str(idx)

    if used == 0:
        raise RuntimeError(f"[no samples used] missing_graph={missing}/{len(samples)}")

    # ---- write top edges per edge type ----
    for et, sc_sum in accum.items():
        src_type, rel, dst_type = et
        sc_avg = sc_sum / max(1, counts[et])

        ei = ref_data[et].edge_index.to(args.device)
        src, dst = ei[0], ei[1]
        m = edge_masks[et]
        if m is not None:
            src = src[m]
            dst = dst[m]

        topk = min(args.top_edges, sc_avg.numel())
        vals, idxs = torch.topk(sc_avg, k=topk, largest=True)

        out_path = os.path.join(args.outdir, f"{src_type}__{rel}__{dst_type}.top.tsv")
        with open(out_path, "w") as f:
            f.write("src_id\tdst_id\tscore\trank\n")
            for rank, j in enumerate(idxs.tolist(), start=1):
                s_id = id_of(src_type, src[j])
                t_id = id_of(dst_type, dst[j])
                f.write(f"{s_id}\t{t_id}\t{float(vals[rank-1]):.6e}\t{rank}\n")

        print(f"✅ saved: {out_path} (n={topk})")

    print(f"✅ done. n_used={used}, missing_graph={missing}")


if __name__ == "__main__":
    main()
