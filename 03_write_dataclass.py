import os
import pickle

import numpy as np
import pandas as pd

from progressbar import progressbar
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as PPE


np.random.seed(12345)

pdb_ids_dict = {
"btd":  pd.read_pickle("../get_data/BindingDB/sequence_to_id_map.pkl").values(),
"dtb": pd.read_pickle("../get_data/DUDE/sequence_to_id_map.pkl").values()
}


def get_dataclass_subset(decoy_style, pdb_id, examples):

    def get_example(example_pair):

        try:
            return list(np.array(feat_dicts[example_pair[0]], dtype=float)), example_pair[1]
        except Exception as e:
            return None, example_pair[1]


    feat_dicts = \
        pd.read_pickle(f"data/feat_dicts/training/{decoy_style}/{pdb_id}.pkl")

    print(decoy_style, pdb_id,
          len(set(example[0] for example in examples) -
              set(feat_dicts.keys())),
          len(examples))

    return [get_example(pair) for pair in examples]


def get_examples(decoy_style, pdb_id):

    with open(f"../docking/data/training/{decoy_style}/id/{pdb_id}") as f:
        example_pairs = [line.strip().split() for line in f.readlines()]

    active_pairs = [pair for pair in example_pairs if pair[1][0] == "A"]

    decoy_pairs = [pair for pair in example_pairs if pair[1][0] == "D"]
    np.random.shuffle(decoy_pairs)

    return {pdb_id: active_pairs + decoy_pairs[:len(active_pairs)]}


def get_examples_dict(decoy_style, direction):

    examples_dict = {}
    for pdb_id in pdb_ids_dict[direction]:
        examples_dict.update(get_examples(decoy_style, pdb_id))

    return examples_dict


def save_dataclass(decoy_style, direction):

    os.system(f"mkdir dataclass/{decoy_style}/ 2>/dev/null")

    np.random.seed(12345)

    '''
    if os.path.isfile(f"dataclass/{decoy_style}/{direction}.pkl"):
        return
    '''

    examples_dict = get_examples_dict(decoy_style, direction)

    dataclass = []
    for pdb_id in progressbar(examples_dict):
        if pdb_id == "3D0E":
            continue
        try:
            dataclass += get_dataclass_subset(decoy_style, pdb_id,
                                              examples_dict[pdb_id])
        except Exception as e:
            print(e, pdb_id)

    np.random.shuffle(dataclass)

    with open(f"dataclass/{decoy_style}/{direction}.pkl", "wb") as f:
        pickle.dump(dataclass, f)


os.system(f"mkdir dataclass/ 2>/dev/null")
if __name__ == "__main__":
    for decoy_style in ("ZS", "DS"):
        for direction in ("btd", "dtb"):
            save_dataclass(decoy_style, direction)

