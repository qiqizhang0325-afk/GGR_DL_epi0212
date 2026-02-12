import re
from dataclasses import dataclass
from typing import Dict, List, Optional

def canonical_chrom_key(chrom: str) -> str:
    """
    Make a canonical key across fasta/gff/vcf.
    Examples:
      Chr2 / chr2 / 2 -> "2"
      Chr01 / 01 -> "1"
      chrM / ChrM / MT -> "MT"
      chrC / ChrC / PT -> "PT"
    """
    c = chrom.strip()

    # common organelle aliases (adjust if your data uses different)
    c_up = c.upper()
    if c_up in {"M", "CHRM", "MT", "MITO", "MITOCHONDRIA"}:
        return "MT"
    if c_up in {"C", "CHRC", "PT", "PLASTID", "CHLOROPLAST"}:
        return "PT"

    # remove chr/Chr prefix
    c = re.sub(r"^(chr|Chr|CHR)", "", c).strip()

    # if purely numeric, remove leading zeros
    if re.fullmatch(r"\d+", c):
        return str(int(c))

    # fallback: use upper-case as key
    return c.upper()


@dataclass
class ChromMapper:
    """Map canonical chrom key -> real names in fasta/gff/vcf."""
    key2fasta: Dict[str, str]
    key2gff: Dict[str, str]
    key2vcf: Dict[str, str]

    @classmethod
    def build(cls, fasta_chroms: List[str], gff_chroms: List[str], vcf_chroms: List[str]) -> "ChromMapper":
        def build_map(chroms: List[str]) -> Dict[str, str]:
            m = {}
            for c in chroms:
                k = canonical_chrom_key(c)
                # keep first seen; if collisions happen, you can assert or log here
                if k not in m:
                    m[k] = c
            return m

        return cls(
            key2fasta=build_map(fasta_chroms),
            key2gff=build_map(gff_chroms),
            key2vcf=build_map(vcf_chroms),
        )

    def fasta_name(self, chrom: str) -> Optional[str]:
        return self.key2fasta.get(canonical_chrom_key(chrom))

    def gff_name(self, chrom: str) -> Optional[str]:
        return self.key2gff.get(canonical_chrom_key(chrom))

    def vcf_name(self, chrom: str) -> Optional[str]:
        return self.key2vcf.get(canonical_chrom_key(chrom))

    def debug_one(self, chrom: str) -> Dict[str, Optional[str]]:
        k = canonical_chrom_key(chrom)
        return {"key": k, "fasta": self.key2fasta.get(k), "gff": self.key2gff.get(k), "vcf": self.key2vcf.get(k)}
