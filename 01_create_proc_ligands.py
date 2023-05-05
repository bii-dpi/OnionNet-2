import os
import warnings
import numpy as np
import pandas as pd

from pickle import dump
from progressbar import progressbar
from rdkit.Chem.rdmolfiles import MolFromMol2Block
from concurrent.futures import ProcessPoolExecutor as PPE


warnings.filterwarnings('ignore')


def get_id_dict(pdb_id):
    ids = load_ids(pdb_id)
    ligand_dict = get_ligand_dict(pdb_id)
    casual_ids = [id_.split(":")[1] for id_ in ids.iloc[:, 1].tolist()]
    ligands = [(ligand_dict[id_], int(id_.startswith("A")))
               for id_ in ids.iloc[:, 0].tolist()]

    return dict(zip(casual_ids, ligands))


def get_mol(indiv, id_dict):
    smiles = id_dict[indiv.split("\n")[1]]
    is_active = np.float(indiv.split("\n")[1][0] == "A")
    curr_mol = MolFromMol2Block(indiv, sanitize=1)

    if curr_mol is not None:
        return {smiles: curr_mol}


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
            ligand_dict.update(get_mol(indiv, id_to_smiles_dict))
        except Exception as e:
            failed += 1
#            print(e)
    print(failed/len(split_mol2))
    with open(f"data/ligand_dicts/{mode}/{decoy_style}/{pdb_id}.pkl", "wb") as f:
        dump(ligand_dict, f)


for mode in ("training", "testing"):
    for decoy_style in ("ZS", "DS"):
        pdb_ids = os.listdir(f"../docking/data/{mode}/{decoy_style}/id/")
        triplets = [(mode, decoy_style, pdb_id)
                    for pdb_id in pdb_ids]

        save_proc_ligands(triplets[1])
        '''
        with PPE() as executor:
            executor.map(save_proc_ligands, triplets)
        '''

