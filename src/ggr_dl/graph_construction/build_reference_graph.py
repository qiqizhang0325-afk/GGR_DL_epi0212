#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build PlantCaduceus reference graph (PyG HeteroData) from reference embeddings + edge files.

Inputs
------
Node embeddings (from 01_embed_reference*.py):
- gene_embeddings.npz    (ids, embeddings)
- region_embeddings.npz  (ids, embeddings, types[optional])
- window_embeddings.npz  (ids, embeddings, window_size[optional])

Edges:
- gene_gene_ppi_edges.tsv          columns: gene1 gene2 combined_score   (usually already directed)
- tf_region_edges.txt              columns: TF  region_id  score [qvalue...]
- region_gene_chip.txt             columns: region_id gene_id [weight]
- region_gene1000_promoter.txt     columns: region_id gene_id [weight]
- re_to_window.tsv                 columns: region_id window_id (has header)

Outputs
-------
A torch.save()'d dict with:
- "data": PyG HeteroData
- "meta": {node_ids + id2idx maps + region_type_map}
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData


# -------------------------
# IO helpers
# -------------------------
def load_npz(npz_path: str):
    """
    Load {ids, embeddings, ...} produced by 01_embed_reference.
    Returns: (ids: list[str], x: torch.FloatTensor [N,D], extra: dict[str, np.ndarray])
    """
    d = np.load(npz_path, allow_pickle=True)
    ids = d["ids"].tolist()
    x = torch.from_numpy(d["embeddings"]).float()
    extra = {k: d[k] for k in d.files if k not in ("ids", "embeddings")}
    return ids, x, extra


def make_id_map(ids: List[str]) -> Dict[str, int]:
    """Map node id string -> contiguous index."""
    return {x: i for i, x in enumerate(ids)}


def _try_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def read_ppi(ppi_path: str):
    """
    Read PPI edges.
    Expected header then lines like: gene1 gene2 combined_score

    Returns: (pairs: list[(g1,g2)], weight: torch.FloatTensor [E])
    """
    pairs = []
    weights = []
    with open(ppi_path, "r") as f:
        _ = f.readline()  # skip header
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            g1, g2 = cols[0].strip(), cols[1].strip()
            w = _try_float(cols[2]) if len(cols) >= 3 else 1.0
            if w is None:
                w = 1.0
            pairs.append((g1, g2))
            weights.append(w)
    return pairs, torch.tensor(weights, dtype=torch.float)


def read_tf_region(tf_region_path: str):
    """
    Read TF->region edges.
    Typical format: TF  region_id  score  qvalue

    Returns: (pairs, weight)
    """
    pairs = []
    weights = []
    with open(tf_region_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            tf, rid = cols[0].strip(), cols[1].strip()
            score = _try_float(cols[2]) if len(cols) >= 3 else 1.0
            if score is None:
                score = 1.0
            pairs.append((tf, rid))
            weights.append(score)
    return pairs, torch.tensor(weights, dtype=torch.float)


def read_region_gene(path: str):
    """
    Read region->gene edges.
    Accepts either:
      region_id gene_id weight
    or
      region_id gene_id
    Returns: (pairs, weight)
    """
    pairs = []
    weights = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            rid, gid = cols[0].strip(), cols[1].strip()
            w = _try_float(cols[2]) if len(cols) >= 3 else 1.0
            if w is None:
                w = 1.0
            pairs.append((rid, gid))
            weights.append(w)
    return pairs, torch.tensor(weights, dtype=torch.float)


def read_re_to_window(path: str):
    """
    Read region->window mapping.
    File produced by 01_embed_reference has a header: region_id  window_id
    """
    pairs = []
    with open(path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            rid, wid = line.rstrip("\n").split("\t")[:2]
            pairs.append((rid.strip(), wid.strip()))
    return pairs


def _build_edge_index(pairs, src_map, dst_map):
    """Convert (src_id, dst_id) pairs to edge_index and keep only valid ids."""
    src = []
    dst = []
    keep = []
    for i, (a, b) in enumerate(pairs):
        if a in src_map and b in dst_map:
            src.append(src_map[a])
            dst.append(dst_map[b])
            keep.append(i)
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long), keep
    return torch.tensor([src, dst], dtype=torch.long), keep


def _subset_weights(w: torch.Tensor, keep_idx: List[int]):
    if w is None:
        return None
    if len(keep_idx) == 0:
        return torch.empty((0,), dtype=torch.float)
    return w[torch.tensor(keep_idx, dtype=torch.long)]


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--gene_npz", required=True)
    ap.add_argument("--region_npz", required=True)
    ap.add_argument("--window_npz", required=True)

    ap.add_argument("--ppi_edges", required=True)
    ap.add_argument("--tf_region_edges", required=True)
    ap.add_argument("--region_gene_chip", required=True)
    ap.add_argument("--region_gene1000_promoter", required=True)
    ap.add_argument("--re_to_window", required=True)

    ap.add_argument("--ppi_already_directed", action="store_true",
                    help="Set if ppi_edges.tsv already contains BOTH directions (recommended for your pipeline).")
    ap.add_argument("--add_reverse_edges", action="store_true",
                    help="Add reverse relations for all non-symmetric edge types (recommended for HGT/GAT on hetero).")

    ap.add_argument("--out_pt", required=True)
    args = ap.parse_args()

    # --- load node embeddings (reference) ---
    gene_ids, gene_x, _ = load_npz(args.gene_npz)
    region_ids, region_x, region_extra = load_npz(args.region_npz)
    window_ids, window_x, _ = load_npz(args.window_npz)

    gene_map = make_id_map(gene_ids)
    region_map = make_id_map(region_ids)
    window_map = make_id_map(window_ids)

    data = HeteroData()

    # --- TF nodes (for motif edges) ---
    # In chr-specific tests, TFs may live on other chromosomes (e.g., AT1G01060),
    # so treating TF as a 'gene' node will drop all motif edges.
    # We therefore create a dedicated 'tf' node type from the TF ids appearing in tf_region_edges.
    tf_ids = []
    with open(args.tf_region_edges, "r") as f:
        for line in f:
            if not line.strip():
                continue
            tf = line.split("\t")[0].strip()
            tf_ids.append(tf)
    # unique, stable order
    tf_ids = sorted(set(tf_ids))

    # initialize tf embeddings:
    # - if TF id is present in gene nodes (full-genome run), reuse gene embedding
    # - otherwise (chr-only run), use mean gene embedding as a neutral prior
    if gene_x.numel() == 0:
        raise ValueError("gene_x is empty; cannot initialize TF embeddings.")
    gene_x_mean = gene_x.mean(dim=0, keepdim=True)  # [1, D]
    tf_x_rows = []
    for tf in tf_ids:
        if tf in gene_map:
            tf_x_rows.append(gene_x[gene_map[tf]].unsqueeze(0))
        else:
            tf_x_rows.append(gene_x_mean)
    tf_x = torch.cat(tf_x_rows, dim=0) if len(tf_x_rows) > 0 else torch.empty((0, gene_x.size(1)))
    tf_map = make_id_map(tf_ids)

    data["tf"].x = tf_x

    data["gene"].x = gene_x
    data["region"].x = region_x
    data["window"].x = window_x

    # region types (optional)
    region_type_map = {}
    if "types" in region_extra:
        types = region_extra["types"].tolist()
        uniq = sorted(set(types))
        region_type_map = {t: i for i, t in enumerate(uniq)}
        data["region"].type = torch.tensor([region_type_map[t] for t in types], dtype=torch.long)

    # --- edges ---
    # (1) gene -[ppi]-> gene
    ppi_pairs, ppi_w = read_ppi(args.ppi_edges)
    ppi_ei, keep = _build_edge_index(ppi_pairs, gene_map, gene_map)
    ppi_w = _subset_weights(ppi_w, keep)

    if (not args.ppi_already_directed) and ppi_ei.size(1) > 0:
        src, dst = ppi_ei[0].tolist(), ppi_ei[1].tolist()
        w = ppi_w.tolist() if ppi_w is not None else [1.0] * len(src)
        src2, dst2, w2 = [], [], []
        for a, b, ww in zip(src, dst, w):
            src2.append(a); dst2.append(b); w2.append(ww)
            src2.append(b); dst2.append(a); w2.append(ww)
        ppi_ei = torch.tensor([src2, dst2], dtype=torch.long)
        ppi_w = torch.tensor(w2, dtype=torch.float)

    data["gene", "ppi", "gene"].edge_index = ppi_ei
    data["gene", "ppi", "gene"].edge_weight = ppi_w

    # (2) gene -[motif]-> region
    tf_reg_pairs, tf_reg_w = read_tf_region(args.tf_region_edges)
    tf_ei, keep = _build_edge_index(tf_reg_pairs, tf_map, region_map)
    tf_reg_w = _subset_weights(tf_reg_w, keep)
    data["tf", "motif", "region"].edge_index = tf_ei
    data["tf", "motif", "region"].edge_weight = tf_reg_w

    # (3) region -[chip]-> gene
    rg_chip_pairs, rg_chip_w = read_region_gene(args.region_gene_chip)
    chip_ei, keep = _build_edge_index(rg_chip_pairs, region_map, gene_map)
    rg_chip_w = _subset_weights(rg_chip_w, keep)
    data["region", "chip", "gene"].edge_index = chip_ei
    data["region", "chip", "gene"].edge_weight = rg_chip_w

    # (5) region -[promoter1000]-> gene
    rg_p1000_pairs, rg_p1000_w = read_region_gene(args.region_gene1000_promoter)
    p1000_ei, keep = _build_edge_index(rg_p1000_pairs, region_map, gene_map)
    rg_p1000_w = _subset_weights(rg_p1000_w, keep)
    data["region", "promoter1000", "gene"].edge_index = p1000_ei
    data["region", "promoter1000", "gene"].edge_weight = rg_p1000_w

    # (6) region -[contained_in]-> window
    rw_pairs = read_re_to_window(args.re_to_window)
    rw_ei, _ = _build_edge_index(rw_pairs, region_map, window_map)
    data["region", "contained_in", "window"].edge_index = rw_ei

    # --- reverse edges ---
    if args.add_reverse_edges:
        if tf_ei.size(1) > 0:
            data["region", "rev_motif", "tf"].edge_index = torch.stack([tf_ei[1], tf_ei[0]], dim=0)
            data["region", "rev_motif", "tf"].edge_weight = tf_reg_w

        if chip_ei.size(1) > 0:
            data["gene", "rev_chip", "region"].edge_index = torch.stack([chip_ei[1], chip_ei[0]], dim=0)
            data["gene", "rev_chip", "region"].edge_weight = rg_chip_w

        if p1000_ei.size(1) > 0:
            data["gene", "rev_promoter1000", "region"].edge_index = torch.stack([p1000_ei[1], p1000_ei[0]], dim=0)
            data["gene", "rev_promoter1000", "region"].edge_weight = rg_p1000_w

        if rw_ei.size(1) > 0:
            data["window", "contains", "region"].edge_index = torch.stack([rw_ei[1], rw_ei[0]], dim=0)

    meta = {
        "gene_ids": gene_ids,
        "region_ids": region_ids,
        "window_ids": window_ids,
        "gene_id2idx": gene_map,
        "region_id2idx": region_map,
        "window_id2idx": window_map,
        "tf_ids": tf_ids,
        "tf_id2idx": tf_map,
        "region_type_map": region_type_map,
    }

    out = {"data": data, "meta": meta}
    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_pt)
    print(f"✅ saved reference graph: {args.out_pt}")
    print(f"  gene nodes   : {len(gene_ids):,}")
    print(f"  region nodes : {len(region_ids):,}")
    print(f"  window nodes : {len(window_ids):,}")
    print(f"  edge types   : {len(data.edge_types)} -> {data.edge_types}")


if __name__ == "__main__":
    main()
