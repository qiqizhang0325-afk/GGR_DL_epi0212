#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert TAIR10 GAF (tair.gaf) to:
  1) gene2go TSV:  ATxGxxxxx \t GO:xxxxxxx
  2) optional go_name TSV

Strictly extract AGI locus IDs.
"""

import argparse
import re
from collections import defaultdict

AGI_RE = re.compile(r"AT[1-5MC]G\d{5}", re.IGNORECASE)


def extract_agi(*fields):
    """
    Try to extract ATxGxxxxx from multiple fields.
    """
    for field in fields:
        if not field:
            continue
        for token in field.replace("|", " ").split():
            m = AGI_RE.match(token.upper())
            if m:
                return m.group(0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaf", required=True)
    ap.add_argument("--out_gene2go", required=True)
    ap.add_argument("--out_go_name", default=None)
    ap.add_argument("--drop_not", action="store_true", help="drop NOT annotations")
    args = ap.parse_args()

    gene2go = defaultdict(set)
    go2name = {}

    with open(args.gaf, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("!"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue

            qualifier = parts[3]
            go_id = parts[4]
            obj_name = parts[9] if len(parts) > 9 else ""
            synonyms = parts[10] if len(parts) > 10 else ""
            db_object_id = parts[1]
            db_object_symbol = parts[2]

            if not go_id.startswith("GO:"):
                continue
            if args.drop_not and "NOT" in (qualifier or ""):
                continue

            gene_id = extract_agi(
                db_object_id,
                db_object_symbol,
                synonyms,
            )
            if gene_id is None:
                continue

            gene2go[gene_id].add(go_id)
            if args.out_go_name and obj_name:
                go2name.setdefault(go_id, obj_name)

    with open(args.out_gene2go, "w") as out:
        for g in sorted(gene2go):
            for go in sorted(gene2go[g]):
                out.write(f"{g}\t{go}\n")

    print(f"✅ gene2go written: {args.out_gene2go}")
    print(f"   genes: {len(gene2go)}")

    if args.out_go_name:
        with open(args.out_go_name, "w") as out:
            for go, name in sorted(go2name.items()):
                out.write(f"{go}\t{name}\n")
        print(f"✅ go_name written: {args.out_go_name}")


if __name__ == "__main__":
    main()


'''
python gaf_to_gene2go_tair10.py \
  --gaf tair.gaf \
  --out_gene2go arabidopsis_gene2go.tsv \
  --drop_not
'''