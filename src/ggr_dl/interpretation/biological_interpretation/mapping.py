import os
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch


def load_pack(reference_pt: str):
    # 兼容 PyTorch FutureWarning：优先用 weights_only=True（如果你保存的pt里不只是state_dict，可能需要回退）
    try:
        return torch.load(reference_pt, map_location="cpu", weights_only=True)
    except TypeError:
        # 老版本 torch 没有 weights_only 参数
        return torch.load(reference_pt, map_location="cpu")
    except Exception:
        # 如果你的 pt 不是纯权重（包含对象），weights_only=True 可能会失败，这里回退保持旧行为
        return torch.load(reference_pt, map_location="cpu")


def decode_tf_in_tf_region_tsv(reference_pt: str, tf_region_in: str, tf_region_out: str) -> str:
    pack = load_pack(reference_pt)
    tf_ids = pack["meta"].get("tf_ids", [])

    df = pd.read_csv(tf_region_in, sep="\t")
    cols = list(df.columns)
    if "src_id" not in cols or "dst_id" not in cols:
        raise ValueError(f"tf_region file must have src_id/dst_id columns. Got: {cols}")

    def is_numeric_series(s: pd.Series) -> bool:
        try:
            _ = s.astype(int)
            return True
        except Exception:
            return False

    if "src_tf" in df.columns:
        df.to_csv(tf_region_out, sep="\t", index=False)
        return tf_region_out

    if is_numeric_series(df["src_id"]):
        if not tf_ids:
            raise ValueError("src_id is numeric but meta['tf_ids'] not found in reference_pt.")
        df["src_tf"] = df["src_id"].astype(int).map(
            lambda i: tf_ids[i] if 0 <= i < len(tf_ids) else f"TF_IDX_OUT_OF_RANGE:{i}"
        )
    else:
        df["src_tf"] = df["src_id"].astype(str)

    df.to_csv(tf_region_out, sep="\t", index=False)
    return tf_region_out


def build_r2g_from_edge(pack, edge_type: Tuple[str, str, str]) -> Dict[int, List[int]]:
    data = pack["data"]
    if edge_type not in data.edge_types:
        return {}

    src_type, rel, dst_type = edge_type
    ei = data[edge_type].edge_index
    r2g = defaultdict(list)

    if src_type == "region" and dst_type == "gene":
        for r_idx, g_idx in zip(ei[0].tolist(), ei[1].tolist()):
            r2g[r_idx].append(g_idx)
        return r2g

    if src_type == "gene" and dst_type == "region":
        for g_idx, r_idx in zip(ei[0].tolist(), ei[1].tolist()):
            r2g[r_idx].append(g_idx)
        return r2g

    return {}


def build_region_to_genes_any_direction(pack, direct_rel: str) -> Tuple[Dict[int, List[int]], str]:
    e1 = ("region", direct_rel, "gene")
    r2g = build_r2g_from_edge(pack, e1)
    if r2g:
        return r2g, f"{e1[0]}__{e1[1]}__{e1[2]}"

    e2 = ("gene", f"rev_{direct_rel}", "region")
    r2g = build_r2g_from_edge(pack, e2)
    if r2g:
        return r2g, f"{e2[0]}__{e2[1]}__{e2[2]} (inverted)"

    return {}, "NONE"


def map_tf_regions_to_targets(reference_pt: str, tf_region_readable_tsv: str, out_prefix: str, top_regions: int = 2000):
    pack = load_pack(reference_pt)
    meta = pack["meta"]
    gene_ids = meta["gene_ids"]
    region_id2idx = meta["region_id2idx"]

    r2g_prom, used_prom = build_region_to_genes_any_direction(pack, "promoter1000")
    r2g_chip, used_chip = build_region_to_genes_any_direction(pack, "chip")

    print(f"[DIAG] promoter mapping edge: {used_prom}, mappings={sum(len(v) for v in r2g_prom.values())}")
    print(f"[DIAG] chip mapping edge: {used_chip}, mappings={sum(len(v) for v in r2g_chip.values())}")

    df = pd.read_csv(tf_region_readable_tsv, sep="\t").head(top_regions).copy()
    if "src_tf" not in df.columns:
        df["src_tf"] = df["src_id"].astype(str)

    df["region_idx"] = df["dst_id"].map(lambda r: region_id2idx.get(r, -1))
    df = df[df["region_idx"] >= 0].copy()

    rows = []
    hit_prom = defaultdict(int)
    hit_all = defaultdict(int)

    for _, row in df.iterrows():
        tf = row["src_tf"]
        region = row["dst_id"]
        r_idx = int(row["region_idx"])
        score = float(row["score"])

        for g_idx in r2g_prom.get(r_idx, []):
            g = gene_ids[g_idx]
            rows.append((tf, region, "promoter1000", g, score))
            hit_prom[g] += 1
            hit_all[g] += 1

        for g_idx in r2g_chip.get(r_idx, []):
            g = gene_ids[g_idx]
            rows.append((tf, region, "chip", g, score))
            hit_all[g] += 1

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    chains_path = out_prefix + ".chains.tsv"
    pd.DataFrame(rows, columns=["tf", "region", "link_type", "target_gene", "tf_region_score"]).to_csv(
        chains_path, sep="\t", index=False
    )

    def write_hits(path: str, hitdict: Dict[str, int]) -> str:
        items = sorted(hitdict.items(), key=lambda x: (-x[1], x[0]))
        with open(path, "w") as f:
            f.write("gene_id\thit_count\n")
            for g, c in items:
                f.write(f"{g}\t{c}\n")
        return path

    prom_path = write_hits(out_prefix + ".target_genes.promoter1000.tsv", hit_prom)
    all_path = write_hits(out_prefix + ".target_genes.promoter1000_plus_chip.tsv", hit_all)

    return chains_path, prom_path, all_path
