#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding reference nodes for the PlantCaduceus reference graph.

Embeds:
- genes: from GFF -> gene embedding (adaptive sliding)
- RE (regulatory elements): from edge-defined regions (Chr:start-end) -> fixed-length embedding per type
- windows: generated around RE midpoints -> container windows (intergenic windows, only where RE exists)

Outputs (in --outdir):
- gene_embeddings.npz        (gene_id -> embedding)
- region_embeddings.npz      (region_id -> embedding, with type)
- window_embeddings.npz      (window_id -> embedding)
- re_to_window.tsv           (RE_id -> window_id)  [for later graph building]
- gene_gene_ppi_edges.tsv    (optional; directed edges written for undirected PPI)
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModelForMaskedLM


# -------------------------
# Utilities
# -------------------------
REGION_RE = re.compile(r"^(Chr[\w]+):(\d+)-(\d+)$")


def parse_region_id(region_id: str):
    """
    Parse region_id of format "ChrX:start-end" and return chrom, start, end (1-based inclusive).
    Example: Chr1:1234-5678 -> ("Chr1", 1234, 5678)
    """
    m = REGION_RE.match(region_id.strip())
    if not m:
        raise ValueError(f"Bad region_id format: {region_id}")
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start > end:
        start, end = end, start
    return chrom, start, end


def safe_seq(genome, chrom: str, start: int, end: int) -> str:
    """Extract sequence from FASTA genome dict with bounds checking (1-based inclusive)."""
    if chrom not in genome:
        return ""
    if start < 1:
        start = 1
    seq_obj = genome[chrom].seq
    if end > len(seq_obj):
        end = len(seq_obj)
    if start > end:
        return ""
    return str(seq_obj[start - 1:end]).upper()


def pad_or_center_truncate(seq: str, fixed_len: int) -> str:
    """Pad with N (both sides) or center-truncate to make sequence length == fixed_len."""
    seq = (seq or "").upper()
    L = len(seq)
    if fixed_len <= 0:
        return seq
    if L == fixed_len:
        return seq
    if L > fixed_len:
        center = L // 2
        half = fixed_len // 2
        start = max(0, center - half)
        end = start + fixed_len
        if end > L:
            end = L
            start = end - fixed_len
        return seq[start:end]
    pad_total = fixed_len - L
    left = pad_total // 2
    right = pad_total - left
    return ("N" * left) + seq + ("N" * right)


def split_windows(seq: str, window_size: int, stride: int):
    """Split sequence into windows with given size and stride. Last window is always end-aligned."""
    seq = (seq or "").upper()
    L = len(seq)
    if L == 0:
        return []
    if L <= window_size:
        return [seq]
    out = []
    for i in range(0, L - window_size + 1, stride):
        out.append(seq[i:i + window_size])
    if (L - window_size) % stride != 0:
        out.append(seq[-window_size:])
    return out


def pool_windows(embs: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Pool window embeddings into a single embedding. Mode: mean|max|first|last."""
    if embs.ndim != 2:
        raise ValueError(f"Expected [n,d], got {embs.shape}")
    mode = (mode or "mean").lower()
    if mode == "mean":
        return embs.mean(axis=0)
    if mode == "max":
        return embs.max(axis=0)
    if mode == "first":
        return embs[0]
    if mode == "last":
        return embs[-1]
    return embs.mean(axis=0)


# -------------------------
# Type priority & region loading
# -------------------------
TYPE_PRIORITY = {
    "promoter_1000": 4,
    "promoter_500": 3,
    "chip_peak": 2,
    "motif_site": 1,
    "re": 0,
}


def upgrade_type(old_t: str, new_t: str) -> str:
    """Upgrade region type using TYPE_PRIORITY."""
    if old_t is None:
        return new_t
    return new_t if TYPE_PRIORITY.get(new_t, 0) > TYPE_PRIORITY.get(old_t, 0) else old_t


def load_re_regions(tf_region_edges, region_gene_chip, region_gene500_promoter, region_gene1000_promoter):
    """
    Collect all RE regions referenced by edge files.

    Returns:
      region_map: rid -> {"chrom_fasta","start","end","type"}
    """
    region_map = {}

    def touch(rid: str, rtype: str):
        chrom, start, end = parse_region_id(rid)
        if rid not in region_map:
            region_map[rid] = {
                "id": rid,
                "chrom_fasta": chrom,
                "start": start,
                "end": end,
                "type": rtype,
            }
        else:
            region_map[rid]["type"] = upgrade_type(region_map[rid].get("type"), rtype)

    # TF -> region (motif)
    with open(tf_region_edges, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            touch(parts[1], "motif_site")

    # region -> gene (chip)
    with open(region_gene_chip, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1:
                continue
            touch(parts[0], "chip_peak")

    # promoter 500
    with open(region_gene500_promoter, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            touch(parts[0], "promoter_500")

    # promoter 1000
    with open(region_gene1000_promoter, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            touch(parts[0], "promoter_1000")

    return region_map


# -------------------------
# Window generation
# -------------------------
def region_to_window_id(region_id: str, window_size: int, stride: int):
    """
    Map a region_id "ChrX:start-end" to a stride-aligned container window around its midpoint.

    Returns window_id: "ChrX:win_start-win_end" (1-based inclusive)
    """
    chrom, start, end = parse_region_id(region_id)
    mid = (start + end) // 2  # 1-based
    window_start = ((mid - 1) // stride) * stride + 1
    window_end = window_start + window_size - 1
    return f"{chrom}:{window_start}-{window_end}"


def build_windows_from_regions(region_ids, window_size: int, stride: int):
    """
    Returns:
      window_ids: set
      re_to_window: list of (re_id, window_id)
    """
    window_ids = set()
    re_to_window = []
    for rid in region_ids:
        wid = region_to_window_id(rid, window_size=window_size, stride=stride)
        window_ids.add(wid)
        re_to_window.append((rid, wid))
    return window_ids, re_to_window


# -------------------------
# Embedding helpers
# -------------------------
def embed_seq_fixed(embedder, seq: str, fixed_len: int, token_pooling: str = "mean"):
    """Pad/center-truncate to fixed_len then embed."""
    s = pad_or_center_truncate(seq, fixed_len=fixed_len)
    return embedder.get_single_embedding(s, pooling_strategy=token_pooling)


def embed_seq_adaptive(
    embedder,
    seq: str,
    short_threshold: int,
    window_size: int,
    stride: int,
    token_pooling: str = "mean",
    window_pooling: str = "mean",
):
    """
    Adaptive embedding for long sequences (genes):
    - short: embed directly
    - long: split into windows and pool
    """
    seq = (seq or "").upper()
    if len(seq) == 0:
        return np.zeros((embedder.embedding_dim,), dtype=np.float32)
    if len(seq) <= short_threshold:
        return embedder.get_single_embedding(seq, pooling_strategy=token_pooling)

    windows = split_windows(seq, window_size=window_size, stride=stride)
    embs = []
    for w in windows:
        e = embedder.get_single_embedding(w, pooling_strategy=token_pooling)
        embs.append(e)
    embs = np.stack(embs, axis=0)  # [n,d]
    return pool_windows(embs, mode=window_pooling)


# -------------------------
# Gene parsing from GFF (minimal; expects "ID=" for gene)
# -------------------------
def iter_genes_from_gff(gff_path: str):
    """
    Yields: (gene_id, chrom, start, end, strand)
    Assumes feature type 'gene' and attributes contains 'ID='
    """
    with open(gff_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, source, ftype, start, end, score, strand, phase, attrs = parts
            if ftype != "gene":
                continue

            gene_id = None
            for item in attrs.split(";"):
                item = item.strip()
                if item.startswith("ID="):
                    gene_id = item.replace("ID=", "")
                    break
            if not gene_id:
                continue
            yield gene_id, chrom, int(start), int(end), strand


# -------------------------
# PPI loading
# -------------------------
def load_ppi_edges(ppi_path: str, gene_set: set, min_score: float = 0.0, score_norm: str = "none"):
    """
    Read undirected PPI edges.
    Expected columns (tab-separated): gene1 gene2 combined_score

    - Skips header if present.
    - Filters edges where gene not in gene_set.
    - Filters by min_score.
    - Returns list of (gene1, gene2, score_float).
    """
    edges = []
    if not ppi_path:
        return edges

    def _norm(s: float) -> float:
        if score_norm == "divide_1000":
            return s / 1000.0
        if score_norm == "log1p_divide_1000":
            return np.log1p(s) / np.log1p(1000.0)
        return s

    with open(ppi_path, "r") as f:
        first = f.readline()
        if not first:
            return edges

        parts = first.rstrip("\n").split("\t")
        is_header = True
        if len(parts) >= 3:
            try:
                float(parts[2])
                is_header = False
            except Exception:
                is_header = True

        lines_iter = ([first] + list(f)) if (not is_header) else f

        for line in lines_iter:
            if not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            g1, g2 = cols[0].strip(), cols[1].strip()
            try:
                score = float(cols[2])
            except Exception:
                continue

            if score < min_score:
                continue
            if (g1 not in gene_set) or (g2 not in gene_set):
                continue
            if g1 == g2:
                continue

            edges.append((g1, g2, _norm(score)))

    return edges


# -------------------------
# Save helpers
# -------------------------
def save_npz_embeddings(path: str, id_list, emb_matrix: np.ndarray, extra: dict | None = None):
    """Store embeddings + ids (+ optional extra arrays) into a compressed NPZ."""
    extra = extra or {}
    np.savez_compressed(path, ids=np.array(id_list, dtype=object), embeddings=emb_matrix.astype(np.float32), **extra)


def save_pairs_tsv(path: str, pairs, header=("src", "dst")):
    """Write pairs into a TSV file with a header."""
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")


# -------------------------
# PlantCAD2 embedder (official-style)
# -------------------------
class PlantCAD2GeneEmbedder:
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 4096):
        self.device = device
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else None,
        )
        self.model.to(device).eval()

        self.config = self.model.config
        self.bidirectional_strategy = getattr(self.config, "bidirectional_strategy", "add")

        # try common config fields
        self.embedding_dim = (
            getattr(self.config, "hidden_size", None)
            or getattr(self.config, "d_model", None)
            or getattr(self.config, "n_embd", None)
            or getattr(self.config, "dim", None)
        )

        # fallback: infer from a tiny forward pass
        if self.embedding_dim is None:
            dummy = "ACGT" * 16
            with torch.no_grad():
                inputs = self.tokenizer(dummy, return_tensors="pt", truncation=True, max_length=256, padding=False)
                input_ids = inputs["input_ids"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                    out = self.model(input_ids=input_ids, output_hidden_states=True)
            self.embedding_dim = int(out.hidden_states[-1].shape[-1])

        self.embedding_dim = int(self.embedding_dim)
        print(f"📏 Embedding dim = {self.embedding_dim}")
        print(f"🔄 Bidirectional strategy = {self.bidirectional_strategy}")
        print(f"🔢 Model max_length = {self.max_length}")

    @staticmethod
    def _revcomp(seq: str) -> str:
        return str(Seq(seq).reverse_complement())

    def _encode_input_ids(self, seq: str) -> torch.Tensor:
        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return inputs["input_ids"].to(self.device)

    @torch.no_grad()
    def get_single_embedding(self, seq: str, pooling_strategy: str = "mean") -> np.ndarray:
        seq = (seq or "").upper()
        if len(seq) == 0:
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        # forward
        input_ids = self._encode_input_ids(seq)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_f = self.model(input_ids=input_ids, output_hidden_states=True)
        h_f = out_f.hidden_states[-1].squeeze(0)  # [L, d]

        # reverse-complement
        rc_seq = self._revcomp(seq)
        input_ids_rc = self._encode_input_ids(rc_seq)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_r = self.model(input_ids=input_ids_rc, output_hidden_states=True)
        h_r = out_r.hidden_states[-1].squeeze(0)  # [L, d]

        ps = (pooling_strategy or "mean").lower()
        if ps == "cls":
            emb_f = h_f[0]
            emb_r = h_r[0]
        else:
            emb_f = h_f.mean(dim=0)
            emb_r = h_r.mean(dim=0)

        strat = (self.bidirectional_strategy or "add").lower()
        if strat in ["mean", "avg", "average"]:
            emb = (emb_f + emb_r) / 2.0
        else:
            emb = emb_f + emb_r

        return emb.detach().float().cpu().numpy()

def build_plantcad2_embedder(model_name: str, device: str | None = None, max_length: int = 4096):
    """Factory for PlantCAD2GeneEmbedder (keeps main() clean)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return PlantCAD2GeneEmbedder(model_name=model_name, device=device, max_length=max_length)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fasta", required=True, help="Reference genome fasta (Chr1/Chr2...)")
    ap.add_argument("--gff", required=True, help="GFF for genes")
    ap.add_argument("--tf_region_edges", required=True)
    ap.add_argument("--region_gene_chip", required=True)
    ap.add_argument("--region_gene500_promoter", required=True)
    ap.add_argument("--region_gene1000_promoter", required=True)
    ap.add_argument("--outdir", required=True)

    # window params (container windows)
    ap.add_argument("--window_size", type=int, default=4000)
    ap.add_argument("--window_stride", type=int, default=2000)

    # gene adaptive
    ap.add_argument("--gene_short_threshold", type=int, default=4096)
    ap.add_argument("--gene_window_size", type=int, default=4096)
    ap.add_argument("--gene_stride", type=int, default=2048)

    # token/window pooling
    ap.add_argument("--token_pooling", type=str, default="mean")
    ap.add_argument("--window_pooling", type=str, default="mean")

    # fixed lens for RE subtypes
    ap.add_argument("--motif_fixed_len", type=int, default=512)
    ap.add_argument("--chip_fixed_len", type=int, default=512)
    ap.add_argument("--promoter500_fixed_len", type=int, default=500)
    ap.add_argument("--promoter1000_fixed_len", type=int, default=1100)

    # PlantCAD2 max_length for tokenizer/model forward
    ap.add_argument("--model_max_length", type=int, default=None,
                    help="Tokenizer/model max_length for PlantCAD2 forward. Default = gene_window_size.")

    # gene-gene PPI (undirected) edges
    ap.add_argument("--gene_gene_ppi", default=None, help="Gene-gene PPI file: gene1 gene2 combined_score (tab)")
    ap.add_argument("--ppi_min_score", type=float, default=0.0, help="Filter PPI edges by min combined_score")
    ap.add_argument(
        "--ppi_score_norm",
        type=str,
        default="none",
        choices=["none", "divide_1000", "log1p_divide_1000"],
        help="Optional normalization for combined_score",
    )

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- load genome ----
    print("Loading genome FASTA...")
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))

    # ---- init PlantCAD2 embedder ----
    HF_MODEL = os.environ.get("PLANTCAD2_MODEL", "kuleshov-group/PlantCAD2-Medium-l48-d1024")
    print(f"Loading PlantCAD2 model: {HF_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_max_len = args.model_max_length or args.gene_window_size
    plantcad_embedder = build_plantcad2_embedder(model_name=HF_MODEL, device=device, max_length=model_max_len)

    # ---- load RE regions from edge files ----
    print("Loading RE regions from edge files...")
    region_map = load_re_regions(
        tf_region_edges=args.tf_region_edges,
        region_gene_chip=args.region_gene_chip,
        region_gene500_promoter=args.region_gene500_promoter,
        region_gene1000_promoter=args.region_gene1000_promoter,
    )
    re_ids = list(region_map.keys())
    print(f"  RE regions: {len(re_ids):,}")

    # ---- build windows from RE ----
    window_ids, re_to_window = build_windows_from_regions(re_ids, window_size=args.window_size, stride=args.window_stride)
    window_ids = sorted(list(window_ids))
    print(f"  Windows (container, only where RE exists): {len(window_ids):,}")

    save_pairs_tsv(os.path.join(args.outdir, "re_to_window.tsv"), re_to_window, header=("region_id", "window_id"))

    # ---- embed RE regions ----
    print("Embedding RE regions...")
    re_embs = []
    re_types = []
    re_id_list = []

    for rid in tqdm(re_ids):
        info = region_map[rid]
        chrom, start, end = info["chrom_fasta"], info["start"], info["end"]
        rtype = info.get("type", "re")

        seq = safe_seq(genome, chrom, start, end)
        if not seq:
            continue

        if rtype == "promoter_500":
            fixed_len = args.promoter500_fixed_len
        elif rtype == "promoter_1000":
            fixed_len = args.promoter1000_fixed_len
        elif rtype == "chip_peak":
            fixed_len = args.chip_fixed_len
        elif rtype == "motif_site":
            fixed_len = args.motif_fixed_len
        else:
            fixed_len = args.chip_fixed_len

        emb = embed_seq_fixed(plantcad_embedder, seq, fixed_len=fixed_len, token_pooling=args.token_pooling)
        re_id_list.append(rid)
        re_types.append(rtype)
        re_embs.append(emb)

    re_embs = (
        np.stack(re_embs, axis=0)
        if len(re_embs)
        else np.zeros((0, plantcad_embedder.embedding_dim), dtype=np.float32)
    )
    save_npz_embeddings(
        os.path.join(args.outdir, "region_embeddings.npz"),
        re_id_list,
        re_embs,
        extra={"types": np.array(re_types, dtype=object)},
    )

    # ---- embed windows ----
    print("Embedding windows ...")
    win_embs = []
    win_id_list = []

    for wid in tqdm(window_ids):
        chrom, start, end = parse_region_id(wid)
        seq = safe_seq(genome, chrom, start, end)
        if not seq:
            continue
        emb = embed_seq_fixed(plantcad_embedder, seq, fixed_len=args.window_size, token_pooling=args.token_pooling)
        win_id_list.append(wid)
        win_embs.append(emb)

    win_embs = (
        np.stack(win_embs, axis=0)
        if len(win_embs)
        else np.zeros((0, plantcad_embedder.embedding_dim), dtype=np.float32)
    )
    save_npz_embeddings(
        os.path.join(args.outdir, "window_embeddings.npz"),
        win_id_list,
        win_embs,
        extra={"window_size": np.array([args.window_size])},
    )

    # ---- embed genes ----
    print("Embedding genes from GFF...")
    gene_ids = []
    gene_embs = []
    genes = list(iter_genes_from_gff(args.gff))

    for gene_id, chrom, start, end, strand in tqdm(genes):
        seq = safe_seq(genome, chrom, start, end)
        if not seq:
            continue
        if strand == "-":
            seq = str(Seq(seq).reverse_complement())

        emb = embed_seq_adaptive(
            plantcad_embedder,
            seq,
            short_threshold=args.gene_short_threshold,
            window_size=args.gene_window_size,
            stride=args.gene_stride,
            token_pooling=args.token_pooling,
            window_pooling=args.window_pooling,
        )
        gene_ids.append(gene_id)
        gene_embs.append(emb)

    gene_embs = (
        np.stack(gene_embs, axis=0)
        if len(gene_embs)
        else np.zeros((0, plantcad_embedder.embedding_dim), dtype=np.float32)
    )
    save_npz_embeddings(os.path.join(args.outdir, "gene_embeddings.npz"), gene_ids, gene_embs)

    # ---- export gene-gene PPI edges (optional) ----
    if args.gene_gene_ppi:
        print("🧩 Loading gene-gene PPI edges...")
        gene_set = set(gene_ids)
        ppi_edges = load_ppi_edges(
            args.gene_gene_ppi,
            gene_set=gene_set,
            min_score=args.ppi_min_score,
            score_norm=args.ppi_score_norm,
        )

        out_ppi = os.path.join(args.outdir, "gene_gene_ppi_edges.tsv")
        with open(out_ppi, "w") as f:
            f.write("gene1\tgene2\tcombined_score\n")
            for g1, g2, s in ppi_edges:
                # write as directed edges for undirected PPI
                f.write(f"{g1}\t{g2}\t{s}\n")
                f.write(f"{g2}\t{g1}\t{s}\n")

        print(f"  PPI undirected edges: {len(ppi_edges):,}")
        print(f"  Directed edges written: {2 * len(ppi_edges):,}")
        print(f"  Saved: {out_ppi}")

    print("Done.")
    print(f"  Saved: {args.outdir}/gene_embeddings.npz")
    print(f"  Saved: {args.outdir}/region_embeddings.npz")
    print(f"  Saved: {args.outdir}/window_embeddings.npz")
    print(f"  Saved: {args.outdir}/re_to_window.tsv")


if __name__ == "__main__":
    main()
