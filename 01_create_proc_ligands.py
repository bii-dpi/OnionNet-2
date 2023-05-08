import os
import numpy as np
import pandas as pd

from pickle import dump
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


def get_lines(pdb_id, indiv, id_dict):

    smiles = id_dict[indiv.split("\n")[1]]
    is_active = float(indiv.split("\n")[1][0] == "A")

    with open(f"tmp_{pdb_id}.mol2", "w") as f:
        f.write(indiv)

    os.system(f"obabel tmp_{pdb_id}.mol2 -O tmp_{pdb_id}.pdb 2>/dev/null")

    with open(f"tmp_{pdb_id}.pdb") as f:
        lines = f.readlines()

    new_lines = ""
    for line in lines:
        if line[:4] in ['ATOM', 'HETA']:
            gro1 = line[:17]
            gro2 = 'LIG '
            gro3 = line[21:]
            new_lines += gro1 + gro2 + gro3

    os.system(f"rm tmp_{pdb_id}.*")

    return {smiles: (new_lines, is_active)}


def get_id_to_smiles_dict(mode, decoy_style, pdb_id):

    with open(f"../docking/data/{mode}/{decoy_style}/id/{pdb_id}") as f:
        dict_ = [line.strip().split() for line in f.readlines()]

    return {pair[1]: pair[0] for pair in dict_}


def save_proc_ligands(triplet):
    mode, decoy_style, pdb_id = triplet

    os.system(f"mkdir data/ligand_dicts/{mode}/ 2>/dev/null")
    os.system(f"mkdir data/ligand_dicts/{mode}/{decoy_style}/ 2>/dev/null")

    id_to_smiles_dict = get_id_to_smiles_dict(mode, decoy_style, pdb_id)

    with open(f"../docking/data/{mode}/{decoy_style}/mol2/{pdb_id}.mol2") as f:
        split_mol2 = f.read().split("@<TRIPOS>MOLECULE")[1:]

    ligand_dict = {}
    failed = 0
    for indiv in progressbar(split_mol2):
        indiv = "@<TRIPOS>MOLECULE\n" + indiv.strip()
        try:
            ligand_dict.update(get_lines(pdb_id, indiv, id_to_smiles_dict))
        except Exception as e:
            failed += 1
            print(e)
    print(failed/len(split_mol2))
    with open(f"data/ligand_dicts/{mode}/{decoy_style}/{pdb_id}.pkl", "wb") as f:
        dump(ligand_dict, f)


os.system(f"mkdir data/ligand_dicts/ 2>/dev/null")
for mode in ("training", "testing"):
    for decoy_style in ("ZS", "DS"):
        pdb_ids = os.listdir(f"../docking/data/{mode}/{decoy_style}/id/")
        triplets = [(mode, decoy_style, pdb_id)
                    for pdb_id in pdb_ids]

        '''
        save_proc_ligands(triplets[1])
        '''
        with PPE() as executor:
            executor.map(save_proc_ligands, triplets)

