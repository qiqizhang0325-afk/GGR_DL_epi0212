# GGR_DL_epi0212

**Graph-Based Deep Learning Framework for Epistatic Interaction Analysis in plant

---

## Overview

This repository provides a reproducible, graph-based deep learning framework for modeling epistatic interactions and gene-level effects in plant.

The method integrates:

* SNP genotype variation
* Regulatory region–gene mapping
* Transcription factor (TF) networks
* Protein–protein interaction (PPI) networks
* Gene Ontology (GO) annotations
* Graph Neural Networks (PyTorch Geometric)

The framework constructs biologically informed heterogeneous graphs and performs downstream gene importance scoring and interaction analysis.

This repository is structured following best practices for reproducible computational genomics research.

---

## Repository Structure

```
GGR_DL_epi0212/
│
├── src/                         # Source code
│   ├── graph_construction/
│   ├── embedding/
│   ├── modeling/
│   ├── interpretation/
│   ├── models.py
│   └── train.py
│
├── data/                        # Input data (NOT version-controlled)
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── scripts/                     # Reproducible execution scripts
│
├── results/                     # Model outputs (not tracked)
│
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
├── .gitignore
└── README.md
```

### Directory Description

| Directory         | Description                                                                        |
| ----------------- | ---------------------------------------------------------------------------------- |
| `src/`            | Core implementation of graph construction, embedding, modeling, and interpretation |
| `data/raw/`       | Original input datasets (VCF, GFF, GAF, etc.)                                      |
| `data/processed/` | Intermediate graph-ready data                                                      |
| `data/external/`  | External reference datasets (e.g., TAIR10)                                         |
| `scripts/`        | Stepwise pipeline execution scripts                                                |
| `results/`        | Model outputs and evaluation results                                               |

Large biological datasets are excluded from version control.

---

## Methodological Pipeline

### Step 1 — Reference Graph Construction

Construct heterogeneous biological graph integrating:

* Gene nodes
* Regulatory region nodes
* TF nodes
* PPI edges
* Promoter-region edges

```bash
python src/graph_construction/build_reference_graph.py
```
```
CUDA_VISIBLE_DEVICES=1 \
PLANTCAD2_MODEL=kuleshov-group/PlantCAD2-Medium-l48-d1024 \
nohup python -m ggr_dl.graph_construction.embed_reference \
  --fasta data/raw/genome/TAIR10_chr2_all.fas \
  --gff data/raw/genome/TAIR10_GFF3_genes.gff \
  --tf_region_edges data/raw/edge_file/chr2_tf_region_edges.txt \
  --region_gene_chip data/raw/edge_file/chr2_region_gene_chip.txt \
  --region_gene1000_promoter data/raw/edge_file/chr2_region_gene1000_promoter.txt \
  --gene_gene_ppi data/raw/edge_file/arabidopsis_PPI_interactions_edge_chr2.txt \
  --ppi_min_score 150 --ppi_score_norm divide_1000 \
  --gene_short_threshold 4096 --gene_window_size 4096 --gene_stride 2048 \
  --outdir ./results/output_embeddings \
  > logs/01_embed_reference.log 2>&1 &
```

```
nohup python -m ggr_dl.graph_construction.build_reference_graph \
  --gene_npz results/output_embeddings/gene_embeddings.npz \
  --region_npz results/output_embeddings/region_embeddings.npz \
  --window_npz results/output_embeddings/window_embeddings.npz \
  --ppi_edges results/output_embeddings/gene_gene_ppi_edges.tsv \
  --tf_region_edges edge_file/chr2_tf_region_edges.txt \
  --region_gene_chip edge_file/chr2_region_gene_chip.txt \
  --region_gene1000_promoter edge_file/chr2_region_gene1000_promoter.txt \
  --re_to_window results/output_embeddings/re_to_window.tsv \
  --ppi_already_directed \
  --add_reverse_edges \
  --out_pt results/reference_graph_chr2.pt \
  > logs/02_build_reference_graph.log 2>&1 &
```


---

### Step 2 — Variant Embedding

Embed SNP delta features into graph structure:

```bash
python src/embedding/embed_sample_variants.py
```
```
python -m ggr_dl.embedding.embed_sample_delta \
  --fasta data/raw/genome/TAIR10_chr2_all.fas \
  --gff data/raw/genome/TAIR10_GFF3_genes.gff \
  --vcf data/raw/genome/arb_ld_chr2.vcf.gz \
  --ref_gene_npz results/output_embeddings/gene_embeddings.npz \
  --ref_region_npz results/output_embeddings/region_embeddings.npz \
  --ref_window_npz results/output_embeddings/window_embeddings.npz \
  --samples 9343_9343 \
  --outdir results/sample_delta/9343_9343

python -m ggr_dl.embedding.build_sample_graph \
  --ref_graph_pt results/reference_graph_chr2.pt \
  --delta_dir results/sample_delta/9343_9343 \
  --outdir results/sample_graphs/9343_9343 \
  --samples 9343_9343 \
  --copy_stats
```

---

### Step 3 — Model Training

Train Graph Neural Network model:

```bash
python src/train.py
```
```
python -m ggr_dl.modeling.train \
  --graphs_dir results/sample_graphs \
  --labels_tsv sample/arb.phe1386.txt \
  --outdir results/train_runs/run1 \
  --epochs 5 \
  --batch_size 1 \
  --train_ratio 0.8 \
  --readout_ntype gene \
  --readout mean
```
---

### Step 4 — Gene Importance Scoring

```bash
python src/interpretation/gene_importance.py
```

```
python -m ggr_dl.analysis.gene_importance \
  --reference_pt results/reference_graph_chr2.pt \
  --checkpoint_pt results/train_runs/run1/checkpoint.pt \
  --graphs_dir results/sample_graphs \
  --sample_list results/train_runs/run1/val_samples.txt \
  --out_prefix results/train_runs/run1/val \
  --device cuda

```
---

### Step 5 — Interaction Scoring

```bash
python src/interpretation/interaction_scoring.py
```

```
python -m ggr_dl.analysis.interaction_scoring \
  --reference_pt results/reference_graph_chr2.pt \
  --checkpoint_pt results/train_runs/run1/checkpoint.pt \
  --graphs_dir results/sample_graphs \
  --gene_importance_tsv results/train_runs/run1/val.gene_importance.tsv \
  --top_genes 2000 \
  --sample_list results/train_runs/run1/val_samples.txt \
  --top_edges 10000 \
  --outdir results/train_runs/run1/interaction_val \
  --device cuda

```

### Step 6 — Biological interpretation (GO / network)
python -m ggr_dl.interpretation.biological_interpretation.cli \
  --reference_pt results/reference_graph_chr2.pt \
  --tf_region_tsv results/train_runs/run1/interaction_val/tf__motif__region.top.tsv \
  --importance_tsv results/train_runs/run1/val.gene_importance.tsv \
  --anno_tsv arabidopsis_gene2go.tsv \
  --run_dir results/train_runs/run1 \
  --top_regions 2000 \
  --top_importance 2000 \
  --min_overlap 5 \
  --min_bg_with_go 5

---

## Environment Setup

### Recommended (Conda)

```bash
conda env create -f environment.yml
conda activate ggr_dl_env
```

### Alternative (pip)

```bash
pip install -r requirements.txt
```

---

## Reproducibility

To fully reproduce the computational environment:

```bash
conda create --name ggr_env --file spec-file.txt
```

All experiments were conducted using:

* Python ≥ 3.8
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas
* SciPy
* scikit-learn
* NetworkX

---

## Input Data Requirements

The framework requires:

* TAIR10 genome annotation (GFF3)
* VCF genotype files
* Gene Ontology annotation (GAF)
* PPI interaction networks
* Regulatory region mapping files

Due to size constraints, raw biological datasets are not included in this repository.

---

## Output

The pipeline generates:

* Trained GNN models
* Gene-level importance scores
* Epistatic interaction scores
* GO enrichment results
* Network-based biological interpretation outputs

---

## Citation

If you use this framework in your research, please cite:

> paper

---

## Author

Qi Zhang, Daniel Probst, Nijveen Harm
Wageningen University & Research
Bioinformatics / Computational Genomics

---

## License

This repository is released for academic research purposes.
Please contact the author for licensing inquiries.
