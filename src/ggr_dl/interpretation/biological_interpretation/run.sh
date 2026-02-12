python -m ggr_dl.interpretation.biological_interpretation.cli \
  --reference_pt result/reference_graph_chr2.pt \
  --tf_region_tsv result/train_runs/run1/interaction_val/tf__motif__region.top.tsv \
  --importance_tsv result/train_runs/run1/val.gene_importance.tsv \
  --anno_tsv arabidopsis_gene2go.tsv \
  --run_dir result/train_runs/run1 \
  --top_regions 2000 \
  --top_importance 2000 \
  --min_overlap 5 \
  --min_bg_with_go 5
