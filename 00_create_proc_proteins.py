import os

from progressbar import progressbar


def save_proc_protein(fname, path):

    with open(path) as f:
        with open(f"data/proc_protein/{fname}", "w") as g:
            for each in f:
                if "ATOM" in each:
                    g.writelines(each)
                    _ = '''
                elif 'HETATM' in each:
                    a.writelines(each)
                        '''
                else:
                    continue


def save_proc_proteins(mode):

    pdb_pairs = [(fname, f"../docking/data/{mode}/pdb/{fname}")
                 for fname in os.listdir(f"../docking/data/{mode}/pdb/")]

    for pair in progressbar(pdb_pairs):
        save_proc_protein(*pair)


os.system("mkdir data/")
os.system("mkdir data/proc_protein/")
for mode in ("training", "testing"):
    save_proc_proteins(mode)

