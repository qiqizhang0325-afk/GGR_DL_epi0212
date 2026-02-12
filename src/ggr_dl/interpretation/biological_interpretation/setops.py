import os
from typing import Dict, Set


def compute_set_ops(promoter_genes_txt: str, all_genes_txt: str, outdir: str) -> Dict[str, str]:
    s_prom = set(open(promoter_genes_txt).read().split())
    s_all = set(open(all_genes_txt).read().split())

    os.makedirs(outdir, exist_ok=True)

    inter = sorted(s_prom & s_all)
    added_by_chip = sorted(s_all - s_prom)

    inter_path = os.path.join(outdir, "intersection_promoter_vs_all.txt")
    add_path = os.path.join(outdir, "genes_added_by_chip.txt")

    with open(inter_path, "w") as f:
        for g in inter:
            f.write(g + "\n")
    with open(add_path, "w") as f:
        for g in added_by_chip:
            f.write(g + "\n")

    return {
        "intersection": inter_path,
        "added_by_chip": add_path,
        "n_promoter": str(len(s_prom)),
        "n_all": str(len(s_all)),
        "n_intersection": str(len(inter)),
        "n_added_by_chip": str(len(added_by_chip)),
    }
