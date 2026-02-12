import os
from typing import Dict, Optional, Set

import pandas as pd
import torch


def write_background_genes(reference_pt: str, out_path: str) -> str:
    # 如果你希望这里也消除 FutureWarning，可以复用 mapping.load_pack；这里先保持最小改动
    pack = torch.load(reference_pt, map_location="cpu")
    gene_ids = pack["meta"]["gene_ids"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for g in gene_ids:
            f.write(g + "\n")
    return out_path


def write_one_col_genes_from_hits(hits_tsv: str, out_txt: str) -> str:
    df = pd.read_csv(hits_tsv, sep="\t")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for g in df["gene_id"].tolist():
            f.write(str(g) + "\n")
    return out_txt


def write_one_col_from_set(s: Set[str], out_txt: str) -> str:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for g in sorted(s):
            f.write(g + "\n")
    return out_txt


def extract_importance_topN(importance_tsv: str, topn: int, out_txt: str) -> str:
    df = pd.read_csv(importance_tsv, sep="\t")
    if "id" not in df.columns:
        raise ValueError(f"importance_tsv must contain column 'id'. Got: {list(df.columns)}")
    genes = df["id"].head(topn).tolist()
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for g in genes:
            f.write(str(g) + "\n")
    return out_txt


def overlap_two_lists(a_txt: str, b_txt: str, out_txt: str) -> str:
    a = set(open(a_txt).read().split())
    b = set(open(b_txt).read().split())
    inter = sorted(a & b)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for g in inter:
            f.write(g + "\n")
    return out_txt
