#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02a_embed_sample_delta_snp_only_fast_chrommap_batched.py

Large-scale version of your 02a script:
- FIX: add missing --promoter500_fixed_len
- FIX: genotype missing handled as -1 (cyvcf2)
- SPEED: VCF querying is interval-based (merged loci intervals), not whole-chrom scan
- SPEED: model inference is batched (forward + reverse-complement in batch)
- SAFETY: optional REF allele check (can warn/skip)
- CLEAN: remove huge debug prints; add --verbose

Outputs are compatible with your current delta-only format:
  - sample_<S>_mutant_gene.npz
  - sample_<S>_mutant_region.npz
  - sample_<S>_mutant_window.npz
  - sample_<S>_stats.json
SNP cache (per sample, per FASTA chrom):
  - snp_cache/snp_<S>_<Chr>.npz  (only SNPs inside merged loci intervals)
"""

import os
import re
import json
import argparse
from pathlib import Path
from bisect import bisect_left
from typing import Dict, List, Tuple, Optional, Iterable

from chrom_mapper import ChromMapper, canonical_chrom_key

import numpy as np
import torch
from tqdm import tqdm
from cyvcf2 import VCF
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModelForMaskedLM

REGION_RE = re.compile(r"^(Chr[\w]+):(\d+)-(\d+)$")


# -------------------------
# Basic IO
# -------------------------
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_sample_id(sample: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in sample)


def load_npz_embeddings(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    ids = d["ids"].tolist()
    embs = d["embeddings"]
    extra = {k: d[k] for k in d.files if k not in ("ids", "embeddings")}
    return ids, embs, extra


def save_npz_embeddings(path: str | Path, ids: List[str], embs: np.ndarray, extra: dict | None = None):
    extra = extra or {}
    np.savez_compressed(str(path), ids=np.array(ids, dtype=object), embeddings=embs.astype(np.float32), **extra)


# -------------------------
# Genomic utils
# -------------------------
def parse_region_id(region_id: str) -> Tuple[str, int, int]:
    m = REGION_RE.match(region_id.strip())
    if not m:
        raise ValueError(f"Bad region_id: {region_id}")
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if start > end:
        start, end = end, start
    return chrom, start, end


def safe_seq(genome: Dict, chrom: str, start: int, end: int) -> str:
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


def reverse_complement(seq: str) -> str:
    return str(Seq(seq).reverse_complement())


def pad_or_center_truncate(seq: str, fixed_len: int) -> str:
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


def split_windows(seq: str, window_size: int, stride: int) -> List[str]:
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
    mode = (mode or "mean").lower()
    if embs.size == 0:
        return embs
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
# GFF gene coords
# -------------------------
def iter_genes_from_gff(gff_path: str, chrom_mapper: ChromMapper | None = None):
    with open(gff_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, ftype, start, end, _, strand, _, attrs = parts
            if chrom_mapper is not None:
                chrom = chrom_mapper.fasta_name(chrom) or chrom
            if ftype != "gene":
                continue
            gid = None
            for item in attrs.split(";"):
                item = item.strip()
                if item.startswith("ID="):
                    gid = item.replace("ID=", "")
                    break
            if gid:
                yield gid, chrom, int(start), int(end), strand


def build_gene_coord_map(gff_path: str, chrom_mapper: ChromMapper | None = None) -> Dict[str, Tuple[str, int, int, str]]:
    m = {}
    for gid, chrom, start, end, strand in iter_genes_from_gff(gff_path, chrom_mapper=chrom_mapper):
        m[gid] = (chrom, start, end, strand)
    return m


def list_gff_chroms(gff_path: str) -> List[str]:
    chroms = set()
    with open(gff_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 1:
                chroms.add(parts[0])
    return sorted(chroms)


# -------------------------
# Locus index + overlap sweep
# -------------------------
class Locus:
    __slots__ = ("id", "chrom", "start", "end", "kind", "strand", "rtype")
    def __init__(self, id: str, chrom: str, start: int, end: int, kind: str,
                 strand: Optional[str] = None, rtype: Optional[str] = None):
        self.id = id
        self.chrom = chrom
        self.start = int(start)
        self.end = int(end)
        self.kind = kind
        self.strand = strand
        self.rtype = rtype


def build_loci_by_chrom(
    gene_ids_ref: List[str],
    gene_coord: Dict[str, Tuple[str, int, int, str]],
    region_ids_ref: List[str],
    region_type_map: Dict[str, str],
    window_ids_ref: List[str],
    gene_flank: int = 0,
) -> Dict[str, List[Locus]]:
    loci_by_chrom: Dict[str, List[Locus]] = {}

    def add(loc: Locus):
        loci_by_chrom.setdefault(loc.chrom, []).append(loc)

    for gid in gene_ids_ref:
        if gid not in gene_coord:
            continue
        chrom, start, end, strand = gene_coord[gid]
        start2 = max(1, start - gene_flank)
        end2 = max(start2, end + gene_flank)
        add(Locus(gid, chrom, start2, end2, kind="gene", strand=strand))

    for rid in region_ids_ref:
        chrom, start, end = parse_region_id(rid)
        rtype = region_type_map.get(rid, "re")
        add(Locus(rid, chrom, start, end, kind="region", rtype=rtype))

    for wid in window_ids_ref:
        chrom, start, end = parse_region_id(wid)
        add(Locus(wid, chrom, start, end, kind="window"))

    for chrom, lst in loci_by_chrom.items():
        lst.sort(key=lambda x: (x.start, x.end))
    return loci_by_chrom


def loci_overlapped_by_snps(loci_sorted: List[Locus], snp_positions_sorted: List[int]) -> List[Locus]:
    if not loci_sorted or not snp_positions_sorted:
        return []
    out = []
    j = 0
    n = len(snp_positions_sorted)
    for loc in loci_sorted:
        while j < n and snp_positions_sorted[j] < loc.start:
            j += 1
        if j < n and snp_positions_sorted[j] <= loc.end:
            out.append(loc)
    return out


def merge_intervals(loci_sorted: List[Locus], gap: int = 0) -> List[Tuple[int, int]]:
    """
    Merge [start,end] intervals of loci on a chrom.
    gap: if next.start <= cur.end + gap => merge (useful to reduce VCF queries)
    """
    if not loci_sorted:
        return []
    ints = [(l.start, l.end) for l in loci_sorted]
    ints.sort()
    merged = []
    cs, ce = ints[0]
    for s, e in ints[1:]:
        if s <= ce + gap:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


# -------------------------
# SNP cache (interval-based)
# -------------------------
def _is_biallelic_snp(rec) -> bool:
    return (len(rec.REF) == 1) and bool(rec.ALT) and (len(rec.ALT[0]) == 1)


def _gt_has_alt(rec, sample_index: int) -> bool:
    """
    cyvcf2: missing allele is often -1, not None.
    Treat presence of allele '1' as ALT.
    """
    a, b = rec.genotypes[sample_index][:2]
    if a is None or b is None:
        return False
    if a < 0 or b < 0:
        return False
    return (a == 1) or (b == 1)


def build_snp_dict_for_sample_chrom_intervals(
    vcf: VCF,
    sample_index: int,
    vcf_chrom: str,
    intervals: List[Tuple[int, int]],
    strict_ref_check: bool = False,
    genome_seq: Optional[str] = None,
    warn_limit: int = 20,
) -> Tuple[List[int], Dict[int, str], Dict[str, int]]:
    """
    Query SNPs only inside merged loci intervals.
    Returns:
      positions_sorted, pos2alt, stats (counts)
    If strict_ref_check=True, genome_seq must be full chrom sequence (uppercase),
    and we skip SNPs where genome_base != rec.REF.
    """
    pos2alt: Dict[int, str] = {}
    stats = {"vcf_records_seen": 0, "snp_biallelic": 0, "gt_alt": 0, "ref_mismatch": 0}
    warned = 0

    for s, e in intervals:
        region = f"{vcf_chrom}:{s}-{e}"
        for rec in vcf(region):
            stats["vcf_records_seen"] += 1
            if not _is_biallelic_snp(rec):
                continue
            stats["snp_biallelic"] += 1
            if not _gt_has_alt(rec, sample_index):
                continue
            stats["gt_alt"] += 1

            p = int(rec.POS)
            alt = str(rec.ALT[0]).upper()

            if strict_ref_check:
                if genome_seq is None or p < 1 or p > len(genome_seq):
                    continue
                ref_base = genome_seq[p - 1]
                if ref_base != str(rec.REF).upper():
                    stats["ref_mismatch"] += 1
                    if warned < warn_limit:
                        # keep it short; avoid huge logs
                        print(f"[WARN] REF mismatch {vcf_chrom}:{p} FASTA={ref_base} VCF={rec.REF} (skip)")
                        warned += 1
                    continue

            pos2alt[p] = alt

    positions = sorted(pos2alt.keys())
    return positions, pos2alt, stats


def load_or_build_snp_cache(cache_dir: Path, sample: str, chrom: str, build_fn):
    """
    Disk cache stores:
      pos: int64 array
      alt: uint8 array (ASCII bytes), aligned with pos by index
    """
    sid = safe_sample_id(sample)
    f = cache_dir / f"snp_{sid}_{chrom}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        pos = d["pos"].astype(np.int64).tolist()
        alt_bytes = d["alt"].astype(np.uint8).tobytes()
        # alt chars aligned with pos by index
        pos2alt = {p: chr(alt_bytes[i]) for i, p in enumerate(pos)}
        return pos, pos2alt, True

    pos, pos2alt, stats = build_fn()
    alt_bytes = bytes([ord(pos2alt[p]) for p in pos])
    np.savez_compressed(
        f,
        pos=np.array(pos, dtype=np.int64),
        alt=np.frombuffer(alt_bytes, dtype=np.uint8),
        **{f"stat_{k}": np.array([v], dtype=np.int64) for k, v in (stats or {}).items()},
    )
    return pos, pos2alt, False


def inject_snps_into_ref(ref_seq: str, start: int, positions_sorted: List[int], pos2alt: Dict[int, str]) -> str:
    """
    Fast SNP injection (still O(#snps_in_interval)).
    """
    if not ref_seq:
        return ""
    end = start + len(ref_seq) - 1
    i = bisect_left(positions_sorted, start)
    if i >= len(positions_sorted):
        return ref_seq
    b = bytearray(ref_seq.encode("ascii", errors="ignore"))
    while i < len(positions_sorted):
        p = positions_sorted[i]
        if p > end:
            break
        rel = p - start
        if 0 <= rel < len(b):
            alt = pos2alt.get(p, None)
            if alt is not None:
                b[rel] = ord(alt)
        i += 1
    return b.decode("ascii").upper()


# -------------------------
# Region subtype -> fixed length
# -------------------------
def fixed_len_by_region_type(rtype: str, promoter500: int, promoter1000: int, chip_len: int, motif_len: int) -> int:
    rtype = (rtype or "re").lower()
    if rtype == "promoter_500":
        return promoter500
    if rtype == "promoter_1000":
        return promoter1000
    if rtype == "chip_peak":
        return chip_len
    if rtype == "motif_site":
        return motif_len
    return chip_len


# -------------------------
# PlantCAD2 embedder (batched)
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

        cfg = self.model.config
        self.bidirectional_strategy = getattr(cfg, "bidirectional_strategy", "add")

        self.embedding_dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "dim", None)
        )
        if self.embedding_dim is None:
            dummy = "ACGT" * 16
            with torch.no_grad():
                inputs = self.tokenizer(dummy, return_tensors="pt", truncation=True, max_length=256, padding=False)
                input_ids = inputs["input_ids"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                    out = self.model(input_ids=input_ids, output_hidden_states=True)
            self.embedding_dim = int(out.hidden_states[-1].shape[-1])

        self.embedding_dim = int(self.embedding_dim)

    @staticmethod
    def _revcomp(seq: str) -> str:
        return str(Seq((seq or "").upper()).reverse_complement())

    def _batch_tokenize(self, seqs: List[str]) -> Dict[str, torch.Tensor]:
        # padding=True to enable batching; truncation keeps <= max_length
        inputs = self.tokenizer(
            seqs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @staticmethod
    def _pool_hidden(h: torch.Tensor, attn: torch.Tensor, pooling_strategy: str) -> torch.Tensor:
        """
        h: [B, T, D], attn: [B, T]
        """
        ps = (pooling_strategy or "mean").lower()
        if ps == "cls":
            return h[:, 0, :]
        # mean over non-pad tokens
        mask = attn.unsqueeze(-1).to(h.dtype)  # [B,T,1]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B,1]
        return (h * mask).sum(dim=1) / denom

    @torch.no_grad()
    def get_batch_embeddings(self, seqs: List[str], pooling_strategy: str = "mean") -> np.ndarray:
        """
        Batched forward + reverse complement, then bidirectional combine.
        seqs: list of strings (already uppercased ok)
        returns: [B, D] float32
        """
        if not seqs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        seqs_u = [(s or "").upper() for s in seqs]
        # handle empty seqs: keep placeholder "N" so tokenizer works, then zero them out later
        empty_mask = np.array([len(s) == 0 for s in seqs_u], dtype=bool)
        seqs_safe = [("N" if len(s) == 0 else s) for s in seqs_u]

        inputs_f = self._batch_tokenize(seqs_safe)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_f = self.model(**inputs_f, output_hidden_states=True)
        h_f = out_f.hidden_states[-1]  # [B,T,D]
        emb_f = self._pool_hidden(h_f, inputs_f.get("attention_mask"), pooling_strategy)

        rc_seqs = [self._revcomp(s) for s in seqs_safe]
        inputs_r = self._batch_tokenize(rc_seqs)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_r = self.model(**inputs_r, output_hidden_states=True)
        h_r = out_r.hidden_states[-1]
        emb_r = self._pool_hidden(h_r, inputs_r.get("attention_mask"), pooling_strategy)

        strat = (self.bidirectional_strategy or "add").lower()
        if strat in ["mean", "avg", "average"]:
            emb = (emb_f + emb_r) / 2.0
        else:
            emb = emb_f + emb_r

        emb = emb.detach().float().cpu().numpy().astype(np.float32)
        if empty_mask.any():
            emb[empty_mask] = 0.0
        return emb


def chunked(iterable: List, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def embed_fixed_batch(embedder: PlantCAD2GeneEmbedder, seqs: List[str], fixed_len: int, token_pooling: str) -> np.ndarray:
    seqs2 = [pad_or_center_truncate(s, fixed_len=fixed_len) for s in seqs]
    return embedder.get_batch_embeddings(seqs2, pooling_strategy=token_pooling)


def embed_adaptive_grouped(
    embedder: PlantCAD2GeneEmbedder,
    seqs: List[str],
    short_threshold: int,
    window_size: int,
    stride: int,
    token_pooling: str,
    window_pooling: str,
    embed_batch_size: int,
) -> np.ndarray:
    """
    seqs: list of gene sequences (already mutated + strand handled)
    returns: [len(seqs), D]
    Strategy:
      - short: embed directly in batches
      - long: split to windows, embed all windows in batches, pool back per gene
    """
    n = len(seqs)
    if n == 0:
        return np.zeros((0, embedder.embedding_dim), dtype=np.float32)

    is_short = [len((s or "")) <= short_threshold for s in seqs]
    out = np.zeros((n, embedder.embedding_dim), dtype=np.float32)

    # short
    short_idx = [i for i, ok in enumerate(is_short) if ok]
    if short_idx:
        short_seqs = [seqs[i] for i in short_idx]
        embs_short = []
        for b in chunked(short_seqs, embed_batch_size):
            embs_short.append(embedder.get_batch_embeddings(b, pooling_strategy=token_pooling))
        embs_short = np.concatenate(embs_short, axis=0)
        for k, i in enumerate(short_idx):
            out[i] = embs_short[k]

    # long => windows
    long_idx = [i for i, ok in enumerate(is_short) if not ok]
    if long_idx:
        win_seqs: List[str] = []
        win_slices: List[Tuple[int, int]] = []  # per gene: (start,end) indices in win_seqs
        for i in long_idx:
            w = split_windows(seqs[i], window_size=window_size, stride=stride)
            a = len(win_seqs)
            win_seqs.extend(w)
            b = len(win_seqs)
            win_slices.append((a, b))

        # embed all windows in batches
        win_embs_parts = []
        for b in chunked(win_seqs, embed_batch_size):
            win_embs_parts.append(embedder.get_batch_embeddings(b, pooling_strategy=token_pooling))
        win_embs = np.concatenate(win_embs_parts, axis=0)

        # pool per gene
        for j, i in enumerate(long_idx):
            a, b = win_slices[j]
            pooled = pool_windows(win_embs[a:b], mode=window_pooling).astype(np.float32)
            out[i] = pooled

    return out.astype(np.float32)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gff", required=True)
    ap.add_argument("--vcf", required=True)

    ap.add_argument("--ref_gene_npz", required=True)
    ap.add_argument("--ref_region_npz", required=True)
    ap.add_argument("--ref_window_npz", required=True)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--snp_cache_dir", default=None)

    ap.add_argument("--model_name", default=os.environ.get("PLANTCAD2_MODEL", "kuleshov-group/PlantCAD2-Medium-l48-d1024"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_max_length", type=int, default=None)

    ap.add_argument("--embed_batch_size", type=int, default=32, help="Model forward batch size (seqs/windows).")
    ap.add_argument("--vcf_interval_gap", type=int, default=0, help="Merge loci intervals if gap<=this (bp). Larger -> fewer VCF queries.")
    ap.add_argument("--strict_ref_check", action="store_true", help="Skip SNPs where FASTA base != VCF REF (slower).")

    ap.add_argument("--gene_short_threshold", type=int, default=4096)
    ap.add_argument("--gene_window_size", type=int, default=4096)
    ap.add_argument("--gene_stride", type=int, default=2048)
    ap.add_argument("--token_pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--window_pooling", default="mean", choices=["mean", "max", "first", "last"])

    ap.add_argument("--gene_flank", type=int, default=0)

    ap.add_argument("--promoter500_fixed_len", type=int, default=600)
    ap.add_argument("--promoter1000_fixed_len", type=int, default=1100)
    ap.add_argument("--chip_fixed_len", type=int, default=512)
    ap.add_argument("--motif_fixed_len", type=int, default=512)

    ap.add_argument("--samples", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.snp_cache_dir) if args.snp_cache_dir else ensure_dir(outdir / "snp_cache")

    if args.verbose:
        print("📥 Loading genome FASTA...")
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))
    fasta_chroms = list(genome.keys())
    if args.verbose:
        print(f"  FASTA chroms: {len(fasta_chroms)}")

    if args.verbose:
        print("📥 Loading VCF...")
    vcf = VCF(args.vcf)
    vcf_chroms = list(vcf.seqnames)
    all_samples = vcf.samples
    sample_index = {s: i for i, s in enumerate(all_samples)}
    if args.verbose:
        print(f"  VCF chroms  : {len(vcf_chroms)}")
        print(f"  VCF samples : {len(all_samples)}")

    if args.verbose:
        print("📥 Scanning GFF chroms...")
    gff_chroms = list_gff_chroms(args.gff)

    if args.verbose:
        print("🧩 Building ChromMapper (FASTA/GFF/VCF)...")
    chrom_mapper = ChromMapper.build(fasta_chroms=fasta_chroms, gff_chroms=gff_chroms, vcf_chroms=vcf_chroms)

    if args.verbose:
        print("📥 Loading gene coords from GFF (mapped to FASTA chrom names)...")
    gene_coord = build_gene_coord_map(args.gff, chrom_mapper=chrom_mapper)

    if args.verbose:
        print("📥 Loading reference node lists...")
    gene_ids_ref, _, _ = load_npz_embeddings(args.ref_gene_npz)
    region_ids_ref, _, region_extra = load_npz_embeddings(args.ref_region_npz)
    window_ids_ref, _, win_extra = load_npz_embeddings(args.ref_window_npz)

    region_types = region_extra.get("types", None)
    if region_types is None:
        region_types = np.array(["re"] * len(region_ids_ref), dtype=object)
    region_type_map = {rid: str(rt) for rid, rt in zip(region_ids_ref, region_types.tolist())}

    window_size = int(win_extra.get("window_size", np.array([0]))[0])
    if window_size <= 0 and window_ids_ref:
        _, s, e = parse_region_id(window_ids_ref[0])
        window_size = e - s + 1
    if window_size <= 0:
        raise ValueError("Cannot infer window_size from window_embeddings.npz")

    if args.verbose:
        print("🧭 Building loci index...")
    loci_by_chrom = build_loci_by_chrom(
        gene_ids_ref=gene_ids_ref,
        gene_coord=gene_coord,
        region_ids_ref=region_ids_ref,
        region_type_map=region_type_map,
        window_ids_ref=window_ids_ref,
        gene_flank=args.gene_flank,
    )

    # pre-merge intervals per chrom (shared across all samples)
    merged_intervals_by_chrom: Dict[str, List[Tuple[int, int]]] = {
        chrom: merge_intervals(loci, gap=args.vcf_interval_gap) for chrom, loci in loci_by_chrom.items()
    }

    if args.samples:
        target_samples = [s for s in args.samples if s in sample_index]
    else:
        target_samples = all_samples

    print(f"🎯 Target samples: {len(target_samples)}")

    model_max_len = args.model_max_length or args.gene_window_size
    print(f"🤗 Loading PlantCAD2: {args.model_name} on {args.device} (max_length={model_max_len})")
    embedder = PlantCAD2GeneEmbedder(args.model_name, device=args.device, max_length=model_max_len)
    print(f"📏 Embedding dim={embedder.embedding_dim} | bidirectional_strategy={embedder.bidirectional_strategy}")
    print(f"⚙️  embed_batch_size={args.embed_batch_size} | vcf_interval_gap={args.vcf_interval_gap} | strict_ref_check={args.strict_ref_check}")

    for sample in target_samples:
        sid = safe_sample_id(sample)
        sidx = sample_index[sample]
        print(f"\n🧬 Sample: {sample}")

        snp_pos_cache: Dict[str, List[int]] = {}
        snp_alt_cache: Dict[str, Dict[int, str]] = {}

        def get_snp_cache(chrom: str):
            # in-memory
            if chrom in snp_pos_cache:
                return snp_pos_cache[chrom], snp_alt_cache[chrom]

            # if chrom absent in FASTA
            if chrom not in genome:
                snp_pos_cache[chrom] = []
                snp_alt_cache[chrom] = {}
                return [], {}

            # map FASTA chrom -> VCF chrom
            vcf_chrom = chrom_mapper.vcf_name(chrom)
            if vcf_chrom is None:
                vcf_chrom = canonical_chrom_key(chrom)

            intervals = merged_intervals_by_chrom.get(chrom, [])
            if not intervals:
                snp_pos_cache[chrom] = []
                snp_alt_cache[chrom] = {}
                return [], {}

            # for strict ref check, grab full chrom sequence once
            chrom_seq = None
            if args.strict_ref_check:
                chrom_seq = str(genome[chrom].seq).upper()

            pos, pos2alt, _ = load_or_build_snp_cache(
                cache_dir=cache_dir,
                sample=sample,
                chrom=chrom,  # cache keyed by FASTA chrom
                build_fn=lambda: build_snp_dict_for_sample_chrom_intervals(
                    vcf,
                    sidx,
                    vcf_chrom=vcf_chrom,
                    intervals=intervals,
                    strict_ref_check=args.strict_ref_check,
                    genome_seq=chrom_seq,
                ),
            )

            snp_pos_cache[chrom] = pos
            snp_alt_cache[chrom] = pos2alt
            return pos, pos2alt

        # Determine mutated loci (overlap with sample SNP positions)
        overlapped: List[Locus] = []
        for chrom, loci in loci_by_chrom.items():
            pos, _ = get_snp_cache(chrom)
            if not pos:
                continue
            overlapped.extend(loci_overlapped_by_snps(loci, pos))

        genes_to_embed = [l for l in overlapped if l.kind == "gene"]
        regions_to_embed = [l for l in overlapped if l.kind == "region"]
        windows_to_embed = [l for l in overlapped if l.kind == "window"]

        # -------------------------
        # genes (adaptive, batched)
        # -------------------------
        mut_gene_ids: List[str] = []
        gene_seqs: List[str] = []
        for loc in genes_to_embed:
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)
            if loc.strand == "-":
                mut = reverse_complement(mut)
            mut_gene_ids.append(loc.id)
            gene_seqs.append(mut)

        if gene_seqs:
            mut_gene_embs = embed_adaptive_grouped(
                embedder,
                gene_seqs,
                short_threshold=args.gene_short_threshold,
                window_size=args.gene_window_size,
                stride=args.gene_stride,
                token_pooling=args.token_pooling,
                window_pooling=args.window_pooling,
                embed_batch_size=args.embed_batch_size,
            )
        else:
            mut_gene_embs = np.zeros((0, embedder.embedding_dim), dtype=np.float32)

        save_npz_embeddings(outdir / f"sample_{sid}_mutant_gene.npz", mut_gene_ids, mut_gene_embs)

        # -------------------------
        # regions (fixed, batched by subtype)
        # -------------------------
        mut_region_ids: List[str] = []
        mut_region_types: List[str] = []
        region_seqs_by_len: Dict[int, List[str]] = {}
        region_meta_by_len: Dict[int, List[Tuple[str, str]]] = {}  # (id, type)

        for loc in regions_to_embed:
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)
            fixed_len = fixed_len_by_region_type(
                loc.rtype or "re",
                promoter500=args.promoter500_fixed_len,
                promoter1000=args.promoter1000_fixed_len,
                chip_len=args.chip_fixed_len,
                motif_len=args.motif_fixed_len,
            )
            region_seqs_by_len.setdefault(fixed_len, []).append(mut)
            region_meta_by_len.setdefault(fixed_len, []).append((loc.id, loc.rtype or "re"))

        # embed per fixed_len group (so we can pad/truncate consistently)
        region_emb_chunks: List[np.ndarray] = []
        region_ids_out: List[str] = []
        region_types_out: List[str] = []

        for fixed_len, seqs in region_seqs_by_len.items():
            metas = region_meta_by_len[fixed_len]
            # batch model inference
            embs_parts = []
            for b in chunked(seqs, args.embed_batch_size):
                embs_parts.append(embed_fixed_batch(embedder, b, fixed_len=fixed_len, token_pooling=args.token_pooling))
            embs = np.concatenate(embs_parts, axis=0) if embs_parts else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
            region_emb_chunks.append(embs)
            region_ids_out.extend([m[0] for m in metas])
            region_types_out.extend([m[1] for m in metas])

        mut_region_embs = np.concatenate(region_emb_chunks, axis=0) if region_emb_chunks else np.zeros((0, embedder.embedding_dim), dtype=np.float32)

        save_npz_embeddings(
            outdir / f"sample_{sid}_mutant_region.npz",
            region_ids_out,
            mut_region_embs,
            extra={"types": np.array(region_types_out, dtype=object)},
        )

        # -------------------------
        # windows (fixed, batched)
        # -------------------------
        mut_window_ids: List[str] = []
        window_seqs: List[str] = []
        for loc in windows_to_embed:
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)
            mut_window_ids.append(loc.id)
            window_seqs.append(mut)

        if window_seqs:
            emb_parts = []
            for b in chunked(window_seqs, args.embed_batch_size):
                emb_parts.append(embed_fixed_batch(embedder, b, fixed_len=window_size, token_pooling=args.token_pooling))
            mut_window_embs = np.concatenate(emb_parts, axis=0)
        else:
            mut_window_embs = np.zeros((0, embedder.embedding_dim), dtype=np.float32)

        save_npz_embeddings(
            outdir / f"sample_{sid}_mutant_window.npz",
            mut_window_ids,
            mut_window_embs,
            extra={"window_size": np.array([window_size], dtype=np.int32)},
        )

        stats = {
            "sample": sample,
            "mut_gene": len(mut_gene_ids),
            "mut_region": len(region_ids_out),
            "mut_window": len(mut_window_ids),
            "window_size": window_size,
            "model_name": args.model_name,
            "model_max_length": model_max_len,
            "token_pooling": args.token_pooling,
            "window_pooling": args.window_pooling,
            "gene_short_threshold": args.gene_short_threshold,
            "gene_window_size": args.gene_window_size,
            "gene_stride": args.gene_stride,
            "gene_flank": args.gene_flank,
            "embed_batch_size": args.embed_batch_size,
            "vcf_interval_gap": args.vcf_interval_gap,
            "strict_ref_check": bool(args.strict_ref_check),
            "note": "delta-only embeddings; SNP-only biallelic ALT[0] applied when GT contains 1; VCF queried by merged loci intervals; model inference batched",
        }
        with open(outdir / f"sample_{sid}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✅ saved delta npz: genes={stats['mut_gene']}, regions={stats['mut_region']}, windows={stats['mut_window']}")

    print("\n🎉 All samples done.")


if __name__ == "__main__":
    main()
