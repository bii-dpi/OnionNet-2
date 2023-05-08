from imports import *


class ComplexDataset(Dataset):
    def __init__(self, dataclass, remove_none):
        self._indices = None
        self.transform = None
        self.dataclass = dataclass
        if remove_none:
            self.dataclass = [pair for pair in dataclass if pair[0]]

    def len(self):
        return len(self.dataclass)

    def get(self, idx):
        graph, label = self.dataclass[idx]

        return graph, np.float(label[0] == "A")


def get_training_dataloader(decoy_style, direction, seed, batch_size):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if direction == "complete":
        dataclass = \
            pd.read_pickle(f"dataclass/{decoy_style}/btd.pkl")
        dataclass += \
            pd.read_pickle(f"dataclass/{decoy_style}/dtb.pkl")
    else:
        dataclass = \
            pd.read_pickle(f"dataclass/{decoy_style}/{direction}.pkl")

    dataset = ComplexDataset(dataclass, True)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_testing_dataloader(decoy_style, direction, pdb_id, batch_size):

    status_dict = {}
    if direction == "complete":
        prefix = "testing"
    else:
        prefix = "training"

    with open(f"../docking/data/{prefix}/{decoy_style}/id/{pdb_id}") as f:
        status_dict = [line.strip().split() for line in f.readlines()]
    #status_dict = {pair[0]: np.float(pair[1][0] == "A") for pair in status_dict}
    status_dict = {pair[0]: pair[1] for pair in status_dict}

    if direction == "complete":
        joined_graphs_dict = \
            pd.read_pickle(f"joined_graph/testing/{decoy_style}/{pdb_id}.pkl")
    else:
        joined_graphs_dict = \
            pd.read_pickle(f"joined_graph/training/{decoy_style}/{pdb_id}.pkl")

    dataclass = [(graph, status_dict[smiles])
                 for smiles, graph in joined_graphs_dict.items()]
    dataset = ComplexDataset(dataclass, False)

    return DataLoader(dataset, batch_size=batch_size)







    parser.add_argument("-shape", type=int, default=[84, 124, 1], nargs="+",
                        help="Input. Reshape the features.")
    parser.add_argument("-n_features", type=int, default=10416,
                        help="Input. The number of features for each complex. \n"
                             "When shells N=62, n_feautes=21*8*62.")

    # load data
    train = pd.read_csv(args.train_file, index_col=0)

    X_train = train.values[:, :args.n_features]

    # Standardize the features
    scaler = preprocessing.StandardScaler()

    X_train_std = scaler.fit_transform(X_train).reshape([-1] + args.shape)
    X_valid_std = scaler.transform(X_valid).reshape([-1] + args.shape)

    y_train = train.pKa.values
    y_valid = valid.pKa.values

    # Callback
    history = model.fit(X_train_std, y_train)

