import torch
import pandas as pd
from collections import defaultdict

ref_pt = "result/reference_graph_chr2.pt"
tf_region_tsv = "result/train_runs/run1/interaction_val/tf__motif__region.top.readable.tsv"
out_prefix = "result/train_runs/run1/interaction_val/tf_region_gene"

top_regions = 2000  # 你可以改小先测试，比如 200

pack = torch.load(ref_pt, map_location="cpu")
data = pack["data"]
meta = pack["meta"]

region_ids = meta["region_ids"]
region_id2idx = meta["region_id2idx"]
gene_ids = meta["gene_ids"]

df = pd.read_csv(tf_region_tsv, sep="\t")
df = df.head(top_regions)

# 取 region idx
df["region_idx"] = df["dst_id"].map(lambda r: region_id2idx.get(r, -1))
df = df[df["region_idx"] >= 0].copy()

# 建 region->gene 映射（从 reference graph 现有边中抽）
def build_region_to_genes(edge_type):
    # edge_type: ("region","promoter1000","gene") or ("region","chip","gene")
    ei = data[edge_type].edge_index
    r2g = defaultdict(list)
    for r_idx, g_idx in zip(ei[0].tolist(), ei[1].tolist()):
        r2g[r_idx].append(g_idx)
    return r2g

r2g_prom = build_region_to_genes(("region","promoter1000","gene")) if ("region","promoter1000","gene") in data.edge_types else {}
r2g_chip = build_region_to_genes(("region","chip","gene")) if ("region","chip","gene") in data.edge_types else {}

rows = []
gene_hit = defaultdict(int)

for _, row in df.iterrows():
    tf = row.get("src_tf", row.get("src_id"))
    region = row["dst_id"]
    r_idx = int(row["region_idx"])
    score = float(row["score"])

    # promoter targets
    for g_idx in r2g_prom.get(r_idx, []):
        g = gene_ids[g_idx]
        rows.append((tf, region, "promoter1000", g, score))
        gene_hit[g] += 1

    # chip targets
    for g_idx in r2g_chip.get(r_idx, []):
        g = gene_ids[g_idx]
        rows.append((tf, region, "chip", g, score))
        gene_hit[g] += 1

out_chain = pd.DataFrame(rows, columns=["tf","region","link_type","target_gene","tf_region_score"])
out_chain.to_csv(out_prefix + ".chains.tsv", sep="\t", index=False)

out_genes = sorted(gene_hit.items(), key=lambda x: (-x[1], x[0]))
with open(out_prefix + ".target_genes.tsv", "w") as f:
    f.write("gene_id\thit_count\n")
    for g, c in out_genes:
        f.write(f"{g}\t{c}\n")

print("wrote:", out_prefix + ".chains.tsv")
print("wrote:", out_prefix + ".target_genes.tsv")
print("n_regions_used:", len(df), "n_chains:", len(out_chain), "n_target_genes:", len(out_genes))
