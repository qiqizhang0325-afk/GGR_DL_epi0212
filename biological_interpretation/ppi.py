import os
from typing import Optional

import pandas as pd


def extract_ppi_endpoint_genes(ppi_tsv: str, out_genes_txt: str, out_edges_txt: Optional[str] = None) -> str:
    df = pd.read_csv(ppi_tsv, sep="\t")
    for c in ("src_id", "dst_id"):
        if c not in df.columns:
            raise ValueError(f"PPI tsv must have columns src_id/dst_id. Got: {list(df.columns)}")

    genes = set(df["src_id"].astype(str).tolist()) | set(df["dst_id"].astype(str).tolist())

    os.makedirs(os.path.dirname(out_genes_txt), exist_ok=True)
    with open(out_genes_txt, "w") as f:
        for g in sorted(genes):
            f.write(g + "\n")

    if out_edges_txt:
        edges = set()
        for s, t in zip(df["src_id"].astype(str), df["dst_id"].astype(str)):
            a, b = (s, t) if s <= t else (t, s)
            edges.add((a, b))
        with open(out_edges_txt, "w") as f:
            f.write("gene_a\tgene_b\n")
            for a, b in sorted(edges):
                f.write(f"{a}\t{b}\n")

    return out_genes_txt
