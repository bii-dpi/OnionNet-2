import os
import itertools
import numpy as np
import pandas as pd
import mdtraj as md

from pickle import dump
from collections import OrderedDict
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor as PPE


class AtomType():
    def __init__(self, fn):
        self.pdb = md.load(fn)
        self.rec_indices = np.array([])
        self.lig_indices = np.array([])
        self.all_pairs = []
        self.atom_pairs = []

        self.masses = np.array([])
        self.atoms_indices = []
        self.residues_indices = []
        self.residues = []
        self.all_ele = np.array([])
        self.lig_ele = np.array([])

        self.xyz = np.array([])
        self.distances = np.array([])

        self.counts_ = np.array([])

    def parsePDB(self):
        top = self.pdb.topology
        residues = [str(residue)[:3] for residue in top.residues]
        residues_cp = residues.copy()

        # number of all groups in the complex except the ligand
        LIG_number = residues.index('LIG')
        self.residues_indices = [i for i in range(top.n_residues) if i != LIG_number]

        # Get the types of atoms in the protein and ligand
        self.rec_indices = top.select('protein')
        self.lig_indices = top.select('resname LIG')
        table, bond = top.to_dataframe()
        self.all_ele = table['element']
        self.lig_ele = table['element'][self.lig_indices]

        H_num = []
        for num, i in enumerate(self.all_ele):
            if i == 'H':
                H_num.append(num)

        # Get the serial number of the atom in each residue or group
        removes = []
        for i in self.residues_indices:
            atoms = top.residue(i).atoms
            each_atoms = [j.index for j in atoms]
            heavy_atoms = [x for x in each_atoms if not x in H_num]

            if len(heavy_atoms) == 0:
                removes.append(i)
            else:
                self.atoms_indices.append(heavy_atoms)

        if len(removes) != 0:
            for i in removes:
                self.residues_indices.remove(i)
        self.residues = [residues_cp[x] for x in self.residues_indices]

        # Get the 3D coordinates for all atoms
        self.xyz = self.pdb.xyz[0]

        return self

    # Calculate the minimum distance between reisdues in the protein and atoms in the ligand
    def compute_distances(self):
        self.parsePDB()
        distances = []

        for r_atom in self.atoms_indices:
            if len(r_atom) == 0:
                continue

            for l_atom in self.lig_indices:
                ds = []
                for i in r_atom:
                    d = np.sqrt(np.sum(np.square(self.xyz[i] - self.xyz[l_atom])))
                    ds.append(d)
                distances.append(min(ds))

        self.distances = np.array(distances)

        return self

    def cutoff_count(self, distances, cutoff):
        self.counts_ = (self.distances <= cutoff) * 1
        return self

# Define all residue types
all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
               'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
def get_residue(residue):
    if residue in all_residues:
        return residue
    else:
        return 'OTH'

# Define all element types
all_elements = ['H', 'C',  'O', 'N', 'P', 'S', 'Hal', 'DU']
Hal = ['F', 'Cl', 'Br', 'I']
def get_elementtype(e):

    if e in all_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'

# all residue-atom combination pairs
keys = ["_".join(x) for x in list(itertools.product(all_residues, all_elements))]

def generate_features(fn, cutoffs):
    cplx = AtomType(fn)
    cplx.compute_distances()

    # Types of the residue and the atom
    new_residue = list(map(get_residue, cplx.residues))
    new_lig = list(map(get_elementtype, cplx.lig_ele))

    # residue-atom pairs
    residues_lig_atoms_combines = ["_".join(x) for x in list(itertools.product(new_residue, new_lig))]

    # calculate the number of contacts in different shells
    counts = []
    onion_counts = []

    for i, cutoff in enumerate(cutoffs):
        cplx.cutoff_count(cplx.distances, cutoff)
        counts_ = cplx.counts_
        if i == 0:
            onion_counts.append(counts_)
        else:
            onion_counts.append(counts_ - counts[-1])
        counts.append(counts_)
    results = []

    for n in range(len(cutoffs)):
        d = OrderedDict()
        d = d.fromkeys(keys, 0)

        for e_e, c in zip(residues_lig_atoms_combines, onion_counts[n]):
            d[e_e] += c
        results += d.values()

    return np.array(results, dtype=float)


def get_features(pdb_id, smiles, lines):
    with open(f"tmp_{pdb_id}.pdb", "w") as f:
        f.write(lines)

    return {smiles: generate_features(f"tmp_{pdb_id}.pdb", ncutoffs)}


def save_proc_ligands(triplet):
    mode, decoy_style, pdb_id = triplet

    os.system(f"mkdir data/feat_dicts/{mode}/ 2>/dev/null")
    os.system(f"mkdir data/feat_dicts/{mode}/{decoy_style}/ 2>/dev/null")

    protein_lines = pd.read_pickle("data/protein_lines.pkl")[pdb_id]
    ligand_lines_dict = \
        pd.read_pickle(f"data/ligand_dicts/{mode}/{decoy_style}/{pdb_id}.pkl")

    feat_dict = {}
    failed = 0
    for smiles, (ligand_lines, is_active) in progressbar(ligand_lines_dict.items()):
        try:
            feat_dict.update(get_features(pdb_id, smiles,
                                            protein_lines + ligand_lines))
        except Exception as e:
            failed += 1
            print(e)
    print(failed/len(ligand_lines_dict))
    with open(f"data/feat_dicts/{mode}/{decoy_style}/{pdb_id}.pkl", "wb") as f:
        dump(feat_dict, f)


outermost = 0.05 * (62 + 1)
ncutoffs = np.linspace(0.1, outermost, 62)
os.system(f"mkdir data/feat_dicts/ 2>/dev/null")
for mode in ("training", "testing"):
    for decoy_style in ("ZS", "DS"):
        pdb_ids = os.listdir(f"../docking/data/{mode}/{decoy_style}/id/")
        triplets = [(mode, decoy_style, pdb_id)
                    for pdb_id in pdb_ids]

        '''
        save_proc_ligands(triplets[1])
        '''
        with PPE(max_workers=10) as executor:
            executor.map(save_proc_ligands, triplets)

    # Need to round features
    #df.to_csv(args.out, float_format='%.1f')

