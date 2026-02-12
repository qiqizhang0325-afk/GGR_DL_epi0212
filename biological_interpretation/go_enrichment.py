#!/usr/bin/env python3
# go_enrichment.py
# GO enrichment via Hypergeometric test + BH-FDR
# Enhancements:
# - Fold enrichment, GeneRatio, BgRatio
# - gene_list supports 1-col (gene_id) or 2-col (gene_id<TAB>weight/hit_count)
# - optional cap on output genes list length

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np


def read_gene_list(path: str) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Read gene list file.
    Supported formats:
      1) one column: gene_id
      2) two columns: gene_id <tab> weight (e.g., hit_count)
    Returns:
      genes (in file order, duplicates allowed)
      weights dict (gene -> weight) if 2nd column is present and numeric; else None
    """
    genes = []
    weights = {}
    has_weight = False

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            g = parts[0]
            genes.append(g)

            if len(parts) >= 2:
                try:
                    w = float(parts[1])
                    weights[g] = max(weights.get(g, 0.0), w)  # keep max if repeated
                    has_weight = True
                except ValueError:
                    # second col not numeric -> treat as no weights
                    pass

    return genes, (weights if has_weight else None)


def read_one_col_set(path: str) -> Set[str]:
    out = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            out.add(line.split()[0])
    return out


def read_anno_tsv(path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    anno_tsv: gene_id <tab> GO:xxxxxxx [optional extra cols ignored]
    """
    gene2go = defaultdict(set)
    go2gene = defaultdict(set)
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            g, go = parts[0], parts[1]
            gene2go[g].add(go)
            go2gene[go].add(g)
    return gene2go, go2gene


def read_go_names(path: str) -> Dict[str, str]:
    """
    go_names: go_id<TAB>name
    """
    go2name = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                go2name[parts[0]] = parts[1]
    return go2name


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """
    Survival function P[X >= k] for Hypergeometric:
      M: population size (background)
      n: number of successes in population (bg genes annotated with this GO)
      N: draws (foreground size)
      k: observed overlap successes
    """
    max_i = min(n, N)
    if k > max_i:
        return 0.0

    denom = log_choose(M, N)
    logs = []
    for i in range(k, max_i + 1):
        logs.append(log_choose(n, i) + log_choose(M - n, N - i) - denom)

    # log-sum-exp
    m = max(logs)
    return float(math.exp(m) * sum(math.exp(x - m) for x in logs))


def bh_fdr(pvals: List[float]) -> List[float]:
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev

    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


def safe_ratio(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


def main():
    ap = argparse.ArgumentParser(description="GO enrichment using hypergeometric test + BH-FDR")
    ap.add_argument("--gene_list", required=True, help="foreground genes: 1 col gene_id or 2 cols gene_id<TAB>weight")
    ap.add_argument("--background", required=True, help="background genes: one gene per line")
    ap.add_argument("--anno_tsv", required=True, help="annotation: gene_id <tab> GO:xxxxxxx (extra cols ignored)")
    ap.add_argument("--go_names", default=None, help="optional: go_id<TAB>name")
    ap.add_argument("--min_overlap", type=int, default=5, help="min overlap genes (k) to report a term")
    ap.add_argument("--min_bg_with_go", type=int, default=5, help="min background genes annotated to GO (n) to consider")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_terms", type=int, default=200, help="how many terms to output (sorted by FDR then p)")
    ap.add_argument("--max_genes_listed", type=int, default=200, help="max overlap genes to print in output (0=print none)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # foreground
    fg_list, fg_weights = read_gene_list(args.gene_list)
    fg_set = set(fg_list)

    # background
    bg_set = read_one_col_set(args.background)

    # intersect foreground with background
    fg_set = fg_set & bg_set

    # annotations
    _, go2gene = read_anno_tsv(args.anno_tsv)
    go2name = read_go_names(args.go_names) if args.go_names else {}

    M = len(bg_set)  # total background
    N = len(fg_set)  # foreground size
    if M == 0 or N == 0:
        out_path = os.path.join(args.outdir, "go_enrichment.tsv")
        with open(out_path, "w") as f:
            f.write(
                "go_id\tgo_name\toverlap_k\tfg_N\tbg_with_go_n\tbg_M\t"
                "gene_ratio\tbg_ratio\tfold_enrichment\tpvalue\tfdr\toverlap_genes\n"
            )
        print(f"Foreground/background empty after intersection. Wrote: {out_path}")
        return

    results = []
    pvals = []

    for go, genes in go2gene.items():
        genes_bg = genes & bg_set
        n = len(genes_bg)
        if n < args.min_bg_with_go:
            continue

        overlap = genes_bg & fg_set
        k = len(overlap)
        if k < args.min_overlap:
            continue

        p = hypergeom_sf(k=k, M=M, n=n, N=N)

        gene_ratio = safe_ratio(k, N)
        bg_ratio = safe_ratio(n, M)
        fold = (gene_ratio / bg_ratio) if bg_ratio > 0 else float("inf")

        # optional: sort overlap genes by weight (if provided), else alphabetic
        overlap_genes = list(overlap)
        if fg_weights:
            overlap_genes.sort(key=lambda g: (-fg_weights.get(g, 0.0), g))
        else:
            overlap_genes.sort()

        if args.max_genes_listed == 0:
            genes_str = ""
        else:
            genes_str = ",".join(overlap_genes[:args.max_genes_listed])

        results.append(
            (
                go,
                go2name.get(go, ""),
                k,
                N,
                n,
                M,
                gene_ratio,
                bg_ratio,
                fold,
                p,
                genes_str,
            )
        )
        pvals.append(p)

    out_path = os.path.join(args.outdir, "go_enrichment.tsv")
    header = (
        "go_id\tgo_name\toverlap_k\tfg_N\tbg_with_go_n\tbg_M\t"
        "gene_ratio\tbg_ratio\tfold_enrichment\tpvalue\tfdr\toverlap_genes\n"
    )

    if not results:
        with open(out_path, "w") as f:
            f.write(header)
        print(f"No terms passed filters. Wrote empty file: {out_path}")
        return

    fdrs = bh_fdr(pvals)

    out_rows = []
    for row, fdr in zip(results, fdrs):
        go, name, k, N, n, M, gene_ratio, bg_ratio, fold, p, genes_str = row
        out_rows.append((go, name, k, N, n, M, gene_ratio, bg_ratio, fold, p, fdr, genes_str))

    out_rows.sort(key=lambda x: (x[10], x[9]))  # fdr then pvalue

    with open(out_path, "w") as f:
        f.write(header)
        for (go, name, k, N, n, M, gene_ratio, bg_ratio, fold, p, fdr, genes_str) in out_rows[: args.top_terms]:
            f.write(
                f"{go}\t{name}\t{k}\t{N}\t{n}\t{M}\t"
                f"{gene_ratio:.6g}\t{bg_ratio:.6g}\t{fold:.6g}\t"
                f"{p:.3e}\t{fdr:.3e}\t{genes_str}\n"
            )

    print(f"✅ saved: {out_path}")
    print(f"Foreground N={N}, Background M={M}, Terms reported={min(args.top_terms, len(out_rows))}")


if __name__ == "__main__":
    main()


'''
python go_enrichment.py \
  --gene_list result/train_runs/run1/top2000_genes.txt \
  --background result/background_genes.txt \
  --anno_tsv arabidopsis_gene2go.tsv \
  --min_overlap 5 \
  --outdir result/train_runs/run1/go_top2000

'''