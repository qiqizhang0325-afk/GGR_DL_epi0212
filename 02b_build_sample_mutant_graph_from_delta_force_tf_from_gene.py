#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02b_build_sample_mutant_graph_from_delta.py

Goal
----
Build per-sample mutant graphs by applying *delta-only* embeddings (from 02a) onto the
reference graph structure (from 00_build_reference_graph_pyg_prom1000_only.py).

This script DOES NOT embed sequences. It only:
  - loads reference graph .pt (contains HeteroData + meta id2idx)
  - loads delta npz for a sample (gene/region/window)
  - overrides node feature matrices x at corresponding indices
  - saves sample_<S>_mutant_graph.pt

Why separate?
-------------
Matches the reference pipeline division:
  embeddings (01)  -> graph assembly (00)
Now:
  delta embeddings (02a) -> per-sample graph assembly (02b)

Inputs
------
--ref_graph_pt: output of 00_build_reference_graph_pyg_prom1000_only.py
--delta_dir: directory containing sample_<S>_mutant_{gene,region,window}.npz
--outdir: where to write sample graphs

You can build graphs for:
- all samples found in delta_dir, or
- a subset via --samples

Output
------
sample_<S>_mutant_graph.pt (torch.save dict {"data": HeteroData, "meta": meta})
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData


def safe_sample_id(sample: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in sample)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_delta_npz(npz_path: Path) -> Tuple[List[str], np.ndarray, Dict]:
    d = np.load(npz_path, allow_pickle=True)
    ids = d["ids"].tolist()
    embs = d["embeddings"].astype(np.float32)
    extra = {k: d[k] for k in d.files if k not in ("ids", "embeddings")}
    return ids, embs, extra


def apply_overrides(
    data: HeteroData,
    meta: Dict,
    node_type: str,
    ids: List[str],
    embs: np.ndarray,
):
    if len(ids) == 0:
        return
    if node_type not in data.node_types:
        return
    if "x" not in data[node_type]:
        return

    id2idx_key = f"{node_type}_id2idx"
    id2idx = meta.get(id2idx_key, {})
    x = data[node_type].x.clone()
    # sanity: embs shape
    if embs.ndim != 2 or embs.shape[0] != len(ids):
        raise ValueError(f"Bad embeddings shape for {node_type}: ids={len(ids)} embs={embs.shape}")

    for i, nid in enumerate(ids):
        idx = id2idx.get(nid, None)
        if idx is None:
            continue
        if 0 <= idx < x.size(0):
            x[idx] = torch.from_numpy(embs[i]).float()
    data[node_type].x = x


def discover_samples(delta_dir: Path) -> List[str]:
    # expects filenames like sample_<S>_mutant_gene.npz
    out = set()
    for p in delta_dir.glob("sample_*_mutant_gene.npz"):
        name = p.name
        s = name[len("sample_"):]
        s = s[: -len("_mutant_gene.npz")]
        out.add(s)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_graph_pt", required=True)
    ap.add_argument("--delta_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--samples", nargs="*", default=None, help="sample IDs (sanitized) to build; default auto-discover from delta_dir")
    ap.add_argument("--copy_stats", action="store_true", help="copy sample_<S>_stats.json into outdir")
    args = ap.parse_args()

    delta_dir = Path(args.delta_dir)
    outdir = ensure_dir(args.outdir)

    ref_obj = torch.load(args.ref_graph_pt, map_location="cpu")
    ref_data: HeteroData = ref_obj["data"]
    meta = ref_obj.get("meta", {})

    samples = args.samples or discover_samples(delta_dir)
    if not samples:
        raise ValueError("No samples found. Provide --samples or ensure delta_dir has sample_*_mutant_gene.npz")

    print(f"🧩 Building mutant graphs for {len(samples)} samples...")

    for sid in samples:
        # sid is sanitized id in filenames
        gene_npz = delta_dir / f"sample_{sid}_mutant_gene.npz"
        region_npz = delta_dir / f"sample_{sid}_mutant_region.npz"
        window_npz = delta_dir / f"sample_{sid}_mutant_window.npz"

        if not gene_npz.exists() or not region_npz.exists() or not window_npz.exists():
            print(f"⚠️ skip {sid}: missing delta npz")
            continue

        data = ref_data.clone()

        gene_ids, gene_embs, _ = load_delta_npz(gene_npz)
        region_ids, region_embs, _ = load_delta_npz(region_npz)
        window_ids, window_embs, _ = load_delta_npz(window_npz)

        apply_overrides(data, meta, "gene", gene_ids, gene_embs)

        
        # ---- TF-MOTIF: force TF node embeddings from gene node embeddings ----
        # We keep a dedicated 'tf' node type for clear regulatory semantics (tf->motif->region...),
        # but for phenotype prediction we want strict per-sample consistency:
        #   tf.x(tf_id) MUST equal gene.x(gene_id) for the corresponding TF gene.
        # This avoids mixing mutant gene features with reference TF features.
        if "tf" in data.node_types:
            tf_id2idx = meta.get("tf_id2idx", {})
            gene_id2idx = meta.get("gene_id2idx", {})

            if not tf_id2idx:
                print("ℹ️  tf node type present but meta['tf_id2idx'] missing/empty; skip TF sync")
            else:
                if "x" not in data["tf"] or "x" not in data["gene"]:
                    print("ℹ️  missing tf.x or gene.x; skip TF sync")
                else:
                    x_tf = data["tf"].x.clone()
                    x_gene = data["gene"].x  # already overridden by this sample's deltas

                    updated_tf = 0
                    for tf_id, tf_idx in tf_id2idx.items():
                        gene_idx = gene_id2idx.get(tf_id, None)
                        if gene_idx is None:
                            raise RuntimeError(
                                f"TF id {tf_id} not found in meta['gene_id2idx'] "
                                f"(reference graph inconsistency)"
                            )
                        if not (0 <= tf_idx < x_tf.size(0)) or not (0 <= gene_idx < x_gene.size(0)):
                            raise RuntimeError(
                                f"Index out of range for TF sync: tf_idx={tf_idx}/{x_tf.size(0)}, "
                                f"gene_idx={gene_idx}/{x_gene.size(0)}"
                            )
                        x_tf[tf_idx] = x_gene[gene_idx]
                        updated_tf += 1

                    data["tf"].x = x_tf
                    print(f"✅ TF synced from gene.x: {updated_tf} / {len(tf_id2idx)}")
        apply_overrides(data, meta, "region", region_ids, region_embs)
        apply_overrides(data, meta, "window", window_ids, window_embs)

        out_pt = outdir / f"sample_{sid}_mutant_graph.pt"
        torch.save({"data": data, "meta": meta}, out_pt)

        if args.copy_stats:
            stats_src = delta_dir / f"sample_{sid}_stats.json"
            if stats_src.exists():
                stats_dst = outdir / stats_src.name
                stats_dst.write_text(stats_src.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"✅ {sid}: saved {out_pt}")

    print("🎉 Done.")


if __name__ == "__main__":
    main()
