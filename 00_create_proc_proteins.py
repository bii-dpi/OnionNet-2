import os

from pickle import dump
from progressbar import progressbar


def get_kept_lines(pdb_id, path):

    kept_lines = ""
    with open(path) as f:
        for line in f:
            if "ATOM" in line:
                kept_lines += line

    return {pdb_id: kept_lines}


def save_proc_proteins(mode):

    pdb_pairs = [(fname[:-4], f"../docking/data/{mode}/pdb/{fname}")
                 for fname in os.listdir(f"../docking/data/{mode}/pdb/")]

    for pair in progressbar(pdb_pairs):
        kept_lines_dict.update(get_kept_lines(*pair))


os.system("mkdir data/")
kept_lines_dict = {}
for mode in ("training", "testing"):
    save_proc_proteins(mode)

with open("data/protein_lines.pkl", "wb") as f:
    dump(kept_lines_dict, f)

