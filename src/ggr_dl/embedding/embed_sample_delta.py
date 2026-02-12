#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02a_embed_sample_delta_snp_only_fast.py

Goal
----
Generate per-sample *delta-only* embeddings for SNP-only mutations, aligned with the reference
embedding logic (PlantCAD2 bidirectional strategy + pooling).

This script DOES NOT build a graph. It only writes:
  - sample_<S>_mutant_gene.npz
  - sample_<S>_mutant_region.npz
  - sample_<S>_mutant_window.npz
  - sample_<S>_stats.json
plus an on-disk SNP cache to accelerate reruns:
  - snp_cache/snp_<S>_<Chr>.npz

Why split from graph-building?
------------------------------
Mirrors your reference pipeline structure:
  (01) embed_reference -> embeddings.npz
  (00) build_reference_graph -> reference_graph.pt
Here:
  (02a) embed_sample_delta -> delta_embeddings.npz
  (02b) build_sample_graph -> mutant_graph.pt (optional, separate)

Assumptions
-----------
- SNP-only biallelic: len(REF)=1 and len(ALT[0])=1. Others ignored (indels, multi-allelic).
- If genotype contains ALT allele (1) => apply ALT[0]. Hets treated as ALT.

Inputs
------
--fasta: reference genome FASTA (chrom names like Chr1, Chr2...)
--gff: gene annotation (gene features with ID=)
--vcf: bgzip+tabix indexed VCF (.vcf.gz)
--ref_gene_npz / --ref_region_npz / --ref_window_npz: reference embedding outputs for node lists (+types/+window_size)

Outputs
-------
Delta-only NPZs store only mutated nodes for each sample (ids + embeddings + optional metadata).
"""

import os
import re
import json
import argparse
from pathlib import Path
from bisect import bisect_left
from typing import Dict, List, Tuple, Optional

# Chrom name harmonization (fasta/gff/vcf)
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
# SNP cache
# -------------------------
def _is_biallelic_snp(rec) -> bool:
    return (len(rec.REF) == 1) and bool(rec.ALT) and (len(rec.ALT[0]) == 1)


def build_snp_dict_for_sample_chrom(vcf: VCF, sample_index: int, vcf_chrom: str, chrom_len: int):
    region = f"{vcf_chrom}:1-{chrom_len}"
    pos2alt: Dict[int, str] = {}
    for rec in vcf(region):
        if not _is_biallelic_snp(rec):
            continue
        a,b = rec.genotypes[sample_index][:2]
        if a < 0 or b < 0: 
            continue
        if (a != 1 and b != 1):
            continue
        if 1 not in [a, b]:
            continue
        pos2alt[int(rec.POS)] = str(rec.ALT[0]).upper()
    positions = sorted(pos2alt.keys())
    return positions, pos2alt


def load_or_build_snp_cache(cache_dir: Path, sample: str, chrom: str, build_fn):
    sid = safe_sample_id(sample)
    f = cache_dir / f"snp_{sid}_{chrom}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        pos = d["pos"].astype(np.int64).tolist()
        alt = d["alt"].astype(np.uint8).tobytes().decode("ascii")
        pos2alt = {p: alt[i] for i, p in enumerate(pos)}
        return pos, pos2alt, True

    pos, pos2alt = build_fn()
    alt_str = "".join(pos2alt[p] for p in pos)
    np.savez_compressed(
        f,
        pos=np.array(pos, dtype=np.int64),
        alt=np.frombuffer(alt_str.encode("ascii"), dtype=np.uint8),
    )
    return pos, pos2alt, False


def inject_snps_into_ref(ref_seq: str, start: int, positions_sorted: List[int], pos2alt: Dict[int, str]) -> str:
    if not ref_seq:
        return ""
    end = start + len(ref_seq) - 1
    i = bisect_left(positions_sorted, start)
    if i >= len(positions_sorted):
        return ref_seq
    seq_list = list(ref_seq)
    while i < len(positions_sorted):
        p = positions_sorted[i]
        if p > end:
            break
        rel = p - start
        if 0 <= rel < len(seq_list):
            seq_list[rel] = pos2alt.get(p, seq_list[rel])
        i += 1
    return "".join(seq_list).upper()


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


# -------------------------
# Region subtype -> fixed length
# -------------------------
def fixed_len_by_region_type(rtype: str, promoter1000: int, chip_len: int, motif_len: int) -> int:
    rtype = (rtype or "re").lower()
    if rtype == "promoter_1000":
        return promoter1000
    if rtype == "chip_peak":
        return chip_len
    if rtype == "motif_site":
        return motif_len
    return chip_len


# -------------------------
# PlantCAD2 embedder (match reference behavior)
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
            # fallback probe
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

        input_ids = self._encode_input_ids(seq)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_f = self.model(input_ids=input_ids, output_hidden_states=True)
        h_f = out_f.hidden_states[-1].squeeze(0)

        rc_seq = self._revcomp(seq)
        input_ids_rc = self._encode_input_ids(rc_seq)
        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            out_r = self.model(input_ids=input_ids_rc, output_hidden_states=True)
        h_r = out_r.hidden_states[-1].squeeze(0)

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

        return emb.detach().float().cpu().numpy().astype(np.float32)


def embed_fixed(embedder: PlantCAD2GeneEmbedder, seq: str, fixed_len: int, token_pooling: str = "mean") -> np.ndarray:
    s = pad_or_center_truncate(seq, fixed_len=fixed_len)
    return embedder.get_single_embedding(s, pooling_strategy=token_pooling)


def embed_adaptive(
    embedder: PlantCAD2GeneEmbedder,
    seq: str,
    short_threshold: int,
    window_size: int,
    stride: int,
    token_pooling: str = "mean",
    window_pooling: str = "mean",
) -> np.ndarray:
    seq = (seq or "").upper()
    if len(seq) == 0:
        return np.zeros((embedder.embedding_dim,), dtype=np.float32)
    if len(seq) <= short_threshold:
        return embedder.get_single_embedding(seq, pooling_strategy=token_pooling)

    windows = split_windows(seq, window_size=window_size, stride=stride)
    embs = [embedder.get_single_embedding(w, pooling_strategy=token_pooling) for w in windows]
    embs = np.stack(embs, axis=0)
    return pool_windows(embs, mode=window_pooling).astype(np.float32)


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

    ap.add_argument("--gene_short_threshold", type=int, default=4096)
    ap.add_argument("--gene_window_size", type=int, default=4096)
    ap.add_argument("--gene_stride", type=int, default=2048)
    ap.add_argument("--token_pooling", default="mean")
    ap.add_argument("--window_pooling", default="mean")

    ap.add_argument("--gene_flank", type=int, default=0)

    ap.add_argument("--promoter1000_fixed_len", type=int, default=1100)
    ap.add_argument("--chip_fixed_len", type=int, default=512)
    ap.add_argument("--motif_fixed_len", type=int, default=512)

    ap.add_argument("--samples", nargs="*", default=None)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    cache_dir = ensure_dir(args.snp_cache_dir) if args.snp_cache_dir else ensure_dir(outdir / "snp_cache")

    print("📥 Loading genome FASTA...")
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))
    fasta_chroms = list(genome.keys())
    print(f"  FASTA chroms: {len(fasta_chroms)}")

    print("📥 Loading VCF...")
    vcf = VCF(args.vcf)
    vcf_chroms = list(vcf.seqnames)
    all_samples = vcf.samples
    sample_index = {s: i for i, s in enumerate(all_samples)}
    print(f"  VCF chroms  : {len(vcf_chroms)}")
    print(f"  VCF samples : {len(all_samples)}")

    print("📥 Scanning GFF chroms...")
    gff_chroms = list_gff_chroms(args.gff)
    print(f"  GFF chroms  : {len(gff_chroms)}")

    print("🧩 Building ChromMapper (FASTA/GFF/VCF)...")
    chrom_mapper = ChromMapper.build(fasta_chroms=fasta_chroms, gff_chroms=gff_chroms, vcf_chroms=vcf_chroms)
    # quick sanity print for Chr2-like key if present
    if "Chr2" in fasta_chroms:
        print("  mapping Chr2:", chrom_mapper.debug_one("Chr2"))

    print("📥 Loading gene coords from GFF (mapped to FASTA chrom names)...")
    gene_coord = build_gene_coord_map(args.gff, chrom_mapper=chrom_mapper)

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

    print("🧭 Building loci index...")
    loci_by_chrom = build_loci_by_chrom(
        gene_ids_ref=gene_ids_ref,
        gene_coord=gene_coord,
        region_ids_ref=region_ids_ref,
        region_type_map=region_type_map,
        window_ids_ref=window_ids_ref,
        gene_flank=args.gene_flank,
    )


    if args.samples:
        target_samples = [s for s in args.samples if s in sample_index]
    else:
        target_samples = all_samples
    print(f"🎯 Target samples: {len(target_samples)}")

    model_max_len = args.model_max_length or args.gene_window_size
    print(f"🤗 Loading PlantCAD2: {args.model_name} on {args.device} (max_length={model_max_len})")
    embedder = PlantCAD2GeneEmbedder(args.model_name, device=args.device, max_length=model_max_len)
    print(f"📏 Embedding dim={embedder.embedding_dim} | bidirectional_strategy={embedder.bidirectional_strategy}")

    for sample in target_samples:
        sid = safe_sample_id(sample)
        sidx = sample_index[sample]
        print(f"\n🧬 Sample: {sample}")

        snp_pos_cache: Dict[str, List[int]] = {}
        snp_alt_cache: Dict[str, Dict[int, str]] = {}

        def get_snp_cache(chrom: str):
            """
            Return cached SNP positions + alt dict for this sample on a given FASTA chrom name (e.g. 'Chr2').

            - chrom: FASTA/GFF naming (graph/internal uses this, e.g. 'Chr2')
            - VCF query uses mapped chrom name (e.g. '2') via chrom_mapper
            """
            # 1) hit cache
            if chrom in snp_pos_cache:
                return snp_pos_cache[chrom], snp_alt_cache[chrom]

            # 2) if this chrom not in FASTA genome, no SNPs for our loci on this chrom
            if chrom not in genome:
                snp_pos_cache[chrom] = []
                snp_alt_cache[chrom] = {}
                return [], {}

            # 3) FASTA length (for region query end)
            chrom_len = len(genome[chrom])

            # 4) map FASTA chrom -> VCF chrom (e.g. Chr2 -> 2)
            vcf_chrom = chrom_mapper.vcf_name(chrom)
            if vcf_chrom is None:
                # fallback: canonical key often works (Chr2 -> "2")
                vcf_chrom = canonical_chrom_key(chrom)

            # 5) build/load cache from disk
            pos, pos2alt, _ = load_or_build_snp_cache(
                cache_dir=cache_dir,
                sample=sample,
                chrom=chrom,  # keep cache file keyed by FASTA chrom (stable with graph ids)
                build_fn=lambda: build_snp_dict_for_sample_chrom(
                    vcf,
                    sidx,
                    vcf_chrom=vcf_chrom,  # IMPORTANT: query VCF using vcf chrom name
                    chrom_len=chrom_len,
                ),
            )

            # 6) store in memory cache
            snp_pos_cache[chrom] = pos
            snp_alt_cache[chrom] = pos2alt
            return pos, pos2alt

        overlapped: List[Locus] = []
        for chrom, loci in loci_by_chrom.items():
            #res = get_snp_cache(chrom)
            #print(chrom, res)
            pos, _ = get_snp_cache(chrom)
            if not pos:
                continue
            overlapped.extend(loci_overlapped_by_snps(loci, pos))

        genes_to_embed = [l for l in overlapped if l.kind == "gene"]
        regions_to_embed = [l for l in overlapped if l.kind == "region"]
        windows_to_embed = [l for l in overlapped if l.kind == "window"]

        # genes
        mut_gene_ids, mut_gene_embs = [], []
        for loc in tqdm(genes_to_embed, desc="mutant genes", leave=False):
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)
            if loc.strand == "-":
                mut = reverse_complement(mut)

            emb = embed_adaptive(
                embedder,
                mut,
                short_threshold=args.gene_short_threshold,
                window_size=args.gene_window_size,
                stride=args.gene_stride,
                token_pooling=args.token_pooling,
                window_pooling=args.window_pooling,
            )
            mut_gene_ids.append(loc.id)
            mut_gene_embs.append(emb)

        mut_gene_embs = np.stack(mut_gene_embs, axis=0) if mut_gene_embs else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
        save_npz_embeddings(outdir / f"sample_{sid}_mutant_gene.npz", mut_gene_ids, mut_gene_embs)

        # regions
        mut_region_ids, mut_region_embs, mut_region_types = [], [], []
        for loc in tqdm(regions_to_embed, desc="mutant regions", leave=False):
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)

            fixed_len = fixed_len_by_region_type(
                loc.rtype or "re",
                promoter1000=args.promoter1000_fixed_len,
                chip_len=args.chip_fixed_len,
                motif_len=args.motif_fixed_len,
            )
            emb = embed_fixed(embedder, mut, fixed_len=fixed_len, token_pooling=args.token_pooling)
            mut_region_ids.append(loc.id)
            mut_region_types.append(loc.rtype or "re")
            mut_region_embs.append(emb)

        mut_region_embs = np.stack(mut_region_embs, axis=0) if mut_region_embs else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
        save_npz_embeddings(
            outdir / f"sample_{sid}_mutant_region.npz",
            mut_region_ids,
            mut_region_embs,
            extra={"types": np.array(mut_region_types, dtype=object)},
        )

        # windows
        mut_window_ids, mut_window_embs = [], []
        for loc in tqdm(windows_to_embed, desc="mutant windows", leave=False):
            pos, pos2alt = get_snp_cache(loc.chrom)
            ref = safe_seq(genome, loc.chrom, loc.start, loc.end)
            if not ref:
                continue
            mut = inject_snps_into_ref(ref, start=loc.start, positions_sorted=pos, pos2alt=pos2alt)

            emb = embed_fixed(embedder, mut, fixed_len=window_size, token_pooling=args.token_pooling)
            mut_window_ids.append(loc.id)
            mut_window_embs.append(emb)

        mut_window_embs = np.stack(mut_window_embs, axis=0) if mut_window_embs else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
        save_npz_embeddings(
            outdir / f"sample_{sid}_mutant_window.npz",
            mut_window_ids,
            mut_window_embs,
            extra={"window_size": np.array([window_size], dtype=np.int32)},
        )

        stats = {
            "sample": sample,
            "mut_gene": len(mut_gene_ids),
            "mut_region": len(mut_region_ids),
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
            "note": "delta-only embeddings; reference unchanged; SNP-only biallelic ALT[0] applied when GT contains 1",
        }
        with open(outdir / f"sample_{sid}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✅ saved delta npz: genes={stats['mut_gene']}, regions={stats['mut_region']}, windows={stats['mut_window']}")

    print("\n🎉 All samples done.")


if __name__ == "__main__":
    main()
