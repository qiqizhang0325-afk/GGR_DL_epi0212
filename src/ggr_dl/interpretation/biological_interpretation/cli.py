import argparse
import os

from .go_enrichment import run_go_enrichment
from .io_utils import (
    extract_importance_topN,
    overlap_two_lists,
    write_background_genes,
    write_one_col_from_set,
    write_one_col_genes_from_hits,
)
from .mapping import decode_tf_in_tf_region_tsv, map_tf_regions_to_targets
from .ppi import extract_ppi_endpoint_genes
from .setops import compute_set_ops


def main():
    ap = argparse.ArgumentParser(
        description="One-stop biological interpretation: TF->region->gene + GO + PPI (robust to edge direction)"
    )
    ap.add_argument("--reference_pt", required=True)
    ap.add_argument("--tf_region_tsv", required=True)
    ap.add_argument("--importance_tsv", required=True)

    ap.add_argument("--anno_tsv", required=True, help="gene2go tsv (e.g. arabidopsis_gene2go.tsv)")
    ap.add_argument("--go_names", default=None, help="optional go_id<TAB>name")

    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--top_regions", type=int, default=2000)
    ap.add_argument("--top_importance", type=int, default=2000)

    ap.add_argument("--ppi_tsv", default=None)

    ap.add_argument("--min_overlap", type=int, default=5)
    ap.add_argument("--min_bg_with_go", type=int, default=5)
    ap.add_argument("--top_terms", type=int, default=200)
    ap.add_argument("--max_genes_listed", type=int, default=200)

    args = ap.parse_args()

    interaction_dir = os.path.join(args.run_dir, "interaction_val")
    os.makedirs(interaction_dir, exist_ok=True)

    # 1) background genes
    bg_path = os.path.join("result", "background_genes.txt")
    if not os.path.exists(bg_path):
        write_background_genes(args.reference_pt, bg_path)
    print(f"[OK] background: {bg_path}")

    # 2) decode TF indices if needed
    tf_region_readable = os.path.join(interaction_dir, "tf__motif__region.top.readable.tsv")
    decode_tf_in_tf_region_tsv(args.reference_pt, args.tf_region_tsv, tf_region_readable)
    print(f"[OK] tf_region readable: {tf_region_readable}")

    # 3) TF->region->gene mapping
    out_prefix = os.path.join(interaction_dir, "tf_region_gene")
    chains_path, prom_hits, all_hits = map_tf_regions_to_targets(
        args.reference_pt, tf_region_readable, out_prefix, top_regions=args.top_regions
    )
    print(f"[OK] chains: {chains_path}")
    print(f"[OK] hits(promoter): {prom_hits}")
    print(f"[OK] hits(all): {all_hits}")

    # 4) gene sets (one-col txt)
    prom_genes_txt = os.path.join(interaction_dir, "genes_tf_promoter_only.txt")
    all_genes_txt = os.path.join(interaction_dir, "genes_tf_promoter_plus_chip.txt")
    write_one_col_genes_from_hits(prom_hits, prom_genes_txt)
    write_one_col_genes_from_hits(all_hits, all_genes_txt)
    print(f"[OK] promoter-only gene list: {prom_genes_txt}")
    print(f"[OK] promoter+chip gene list: {all_genes_txt}")

    # 5) set ops promoter vs all
    setops_dir = os.path.join(interaction_dir, "setops")
    ops = compute_set_ops(prom_genes_txt, all_genes_txt, setops_dir)
    print(
        f"[OK] promoter_only={ops['n_promoter']} all={ops['n_all']} "
        f"intersection={ops['n_intersection']} added_by_chip={ops['n_added_by_chip']}"
    )

    # 6) importance topN + overlaps
    imp_top_txt = os.path.join(args.run_dir, f"top{args.top_importance}_importance_genes.txt")
    extract_importance_topN(args.importance_tsv, args.top_importance, imp_top_txt)
    print(f"[OK] importance top list: {imp_top_txt}")

    overlap_prom = os.path.join(args.run_dir, "overlap_importance_vs_tf_promoter.txt")
    overlap_all = os.path.join(args.run_dir, "overlap_importance_vs_tf_all.txt")
    overlap_two_lists(imp_top_txt, prom_genes_txt, overlap_prom)
    overlap_two_lists(imp_top_txt, all_genes_txt, overlap_all)
    print(f"[OK] overlap (importance ∩ promoter targets): {overlap_prom}")
    print(f"[OK] overlap (importance ∩ all targets): {overlap_all}")

    # 7) PPI module (optional)
    ppi_tsv = args.ppi_tsv or os.path.join(interaction_dir, "gene__ppi__gene.top.tsv")
    ppi_genes_txt = os.path.join(interaction_dir, "genes_ppi_endpoints.txt")
    ppi_edges_txt = os.path.join(interaction_dir, "ppi_edges_undirected.tsv")

    if os.path.exists(ppi_tsv):
        extract_ppi_endpoint_genes(ppi_tsv, ppi_genes_txt, out_edges_txt=ppi_edges_txt)
        print(f"[OK] PPI endpoints: {ppi_genes_txt}")
        print(f"[OK] PPI undirected edges: {ppi_edges_txt}")

        s_ppi = set(open(ppi_genes_txt).read().split())
        s_prom = set(open(prom_genes_txt).read().split())
        s_all = set(open(all_genes_txt).read().split())
        s_imp = set(open(imp_top_txt).read().split())

        ppi_x_imp = write_one_col_from_set(s_ppi & s_imp, os.path.join(args.run_dir, "overlap_ppi_vs_importance.txt"))
        ppi_x_prom = write_one_col_from_set(s_ppi & s_prom, os.path.join(args.run_dir, "overlap_ppi_vs_tf_promoter.txt"))
        ppi_x_all = write_one_col_from_set(s_ppi & s_all, os.path.join(args.run_dir, "overlap_ppi_vs_tf_all.txt"))
        triple_prom = write_one_col_from_set(s_ppi & s_prom & s_imp, os.path.join(args.run_dir, "triple_ppi_tf_promoter_importance.txt"))
        triple_all = write_one_col_from_set(s_ppi & s_all & s_imp, os.path.join(args.run_dir, "triple_ppi_tf_all_importance.txt"))

        print(f"[OK] overlap PPI∩importance: {ppi_x_imp} (n={len(s_ppi & s_imp)})")
        print(f"[OK] overlap PPI∩TF(promoter): {ppi_x_prom} (n={len(s_ppi & s_prom)})")
        print(f"[OK] overlap PPI∩TF(all): {ppi_x_all} (n={len(s_ppi & s_all)})")
        print(f"[OK] triple PPI∩TF(all)∩importance: {triple_all} (n={len(s_ppi & s_all & s_imp)})")
    else:
        print(f"[WARN] PPI tsv not found, skip PPI integration: {ppi_tsv}")
        ppi_genes_txt = None

    # 8) GO enrichment
    go_out_prom = os.path.join(args.run_dir, "go_tf_promoter_only")
    go_out_all = os.path.join(args.run_dir, "go_tf_promoter_plus_chip")
    go_out_added = os.path.join(args.run_dir, "go_genes_added_by_chip")

    prom_go = run_go_enrichment(
        gene_list_path=prom_hits,
        background_path=bg_path,
        anno_tsv=args.anno_tsv,
        outdir=go_out_prom,
        go_names=args.go_names,
        min_overlap=args.min_overlap,
        min_bg_with_go=args.min_bg_with_go,
        top_terms=args.top_terms,
        max_genes_listed=args.max_genes_listed,
    )
    all_go = run_go_enrichment(
        gene_list_path=all_hits,
        background_path=bg_path,
        anno_tsv=args.anno_tsv,
        outdir=go_out_all,
        go_names=args.go_names,
        min_overlap=args.min_overlap,
        min_bg_with_go=args.min_bg_with_go,
        top_terms=args.top_terms,
        max_genes_listed=args.max_genes_listed,
    )

    added_genes_txt = ops["added_by_chip"]
    added_go = run_go_enrichment(
        gene_list_path=added_genes_txt,
        background_path=bg_path,
        anno_tsv=args.anno_tsv,
        outdir=go_out_added,
        go_names=args.go_names,
        min_overlap=max(2, min(args.min_overlap, 5)),
        min_bg_with_go=args.min_bg_with_go,
        top_terms=args.top_terms,
        max_genes_listed=args.max_genes_listed,
    )

    if ppi_genes_txt:
        go_ppi = run_go_enrichment(
            gene_list_path=ppi_genes_txt,
            background_path=bg_path,
            anno_tsv=args.anno_tsv,
            outdir=os.path.join(args.run_dir, "go_ppi_endpoints"),
            go_names=args.go_names,
            min_overlap=max(2, args.min_overlap),
            min_bg_with_go=args.min_bg_with_go,
            top_terms=args.top_terms,
            max_genes_listed=args.max_genes_listed,
        )
    else:
        go_ppi = None

    print("\n===== SUMMARY OUTPUTS =====")
    print("background:", bg_path)
    print("TF->region readable:", tf_region_readable)
    print("TF->region->gene chains:", chains_path)
    print("Promoter hits:", prom_hits)
    print("All hits:", all_hits)
    print("GO promoter-only:", prom_go)
    print("GO promoter+chip:", all_go)
    print("GO chip-added:", added_go)
    if go_ppi:
        print("GO PPI:", go_ppi)
    print("===========================\n")


if __name__ == "__main__":
    main()
