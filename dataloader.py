from imports import *


class ComplexDataset(Dataset):
    def __init__(self, features, labels):
        self._indices = None
        self.transform = None
        self.features = features
        self.labels = labels

        print(len(labels))

    def len(self):
        return len(self.labels)

    def __len__(self):
        return len(self.labels)

    #def get(self, idx):
    def __getitem__(self, idx):

       # print(self.features[idx][0].shape)
        features = torch.from_numpy(self.features[idx]).float()


        return features, np.float(self.labels[idx] == "A")


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

    dataclass = [pair for pair in dataclass if pair[0]]
    features, labels = zip(*dataclass)

    scaler = StandardScaler()
    #features = scaler.fit_transform(np.array(features)).reshape([-1, 84, 124, 1])
    features = scaler.fit_transform(np.array(features)).reshape([-1, 1, 84, 124])

    with open(f"scalers/{decoy_style}_{direction}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    dataset = ComplexDataset(features, labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_testing_dataloader(decoy_style, direction, pdb_id, batch_size):

    scaler = pd.read_pickle(f"scalers/{decoy_style}_{direction}.pkl")

    status_dict = {}
    if direction == "complete":
        prefix = "testing"
    else:
        prefix = "training"

    with open(f"../docking/data/{prefix}/{decoy_style}/id/{pdb_id}") as f:
        status_dict = [line.strip().split() for line in f.readlines()]
    #status_dict = {pair[0]: np.float(pair[1][0] == "A") for pair in status_dict}
    status_dict = {pair[0]: pair[1][0] for pair in status_dict}

    if direction == "complete":
        joined_graphs_dict = \
            pd.read_pickle(f"data/feat_dicts/testing/{decoy_style}/{pdb_id}.pkl")
    else:
        joined_graphs_dict = \
            pd.read_pickle(f"data/feat_dicts/training/{decoy_style}/{pdb_id}.pkl")

    dataclass = [(scaler.transform(graph.reshape(1, -1)).reshape([-1, 1, 84, 124]),
                  status_dict[smiles])
                 for smiles, graph in joined_graphs_dict.items()]
    features_, labels = zip(*dataclass)
    features = []
    for features_indiv in features_:
        features.append(features_indiv[0])
    features = np.array(features)

    dataset = ComplexDataset(features, labels)

    return DataLoader(dataset, batch_size=batch_size)

