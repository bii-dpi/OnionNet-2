from imports import *
from model import Classifier
from performance import get_performance
from dataloader import get_testing_dataloader


BATCH_SIZE = 64

legal_pdb_ids = \
    list(pd.read_pickle("../get_data/BindingDB/sequence_to_id_map.pkl").values())[:20]
legal_pdb_ids += \
    list(pd.read_pickle("../get_data/DUDE/sequence_to_id_map.pkl").values())[:20]
legal_pdb_ids += \
    list(pd.read_pickle("../ChEMBL/sequence_to_id_map.pkl").values())[:20]
legal_pdb_ids = [pdb_id for pdb_id in legal_pdb_ids if pdb_id not in ["3D0E"]]
print(len(legal_pdb_ids))


# XXX: Need to fix the ChEMBL sequence to ID map.
pdb_ids_dict = {
"dtb": pd.read_pickle("../get_data/BindingDB/sequence_to_id_map.pkl").values(),
"btd": pd.read_pickle("../get_data/DUDE/sequence_to_id_map.pkl").values(),
"complete": os.listdir(f"../docking/data/testing/DS/id/")
}


def write_eval(decoy_style, direction, args):
    device = torch.device(f"cuda:{args['c']}")

    rows = [",".join(["pdb_id", "seed", "epoch",
                      "AUC", "AUPR", "LogAUC", "MCC", "recall_1", "recall_5",
                      "recall_10", "recall_25", "recall_50", "EF_1",
                      "EF_5", "EF_10", "EF_25", "EF_50"])]

    '''
    if os.path.isfile(f"results/{decoy_style}_{direction}.csv"):
        return
    '''

    for pdb_id in progressbar(pdb_ids_dict[direction]):
        try:
            '''
            if pdb_id not in legal_pdb_ids:
                continue
            '''
            curr_testing_dl = \
                get_testing_dataloader(decoy_style, direction, pdb_id, BATCH_SIZE)
            for seed in range(NUM_SEEDS):
#                for epoch in (10, 25, 50, 75, 99):
                for epoch in (49,):
    #            for epoch in (1, 2,):
                    from torchvision.models import resnet50 as resnet
                    from torchvision.models.squeezenet import SqueezeNet as resnet

                    curr_classifier = resnet(num_classes=1).to(device)
                    curr_classifier = Classifier().to(device)
                    curr_path = f"models/{decoy_style}/{direction}/{seed}_{epoch}.pt"
                    curr_classifier.load_state_dict(torch.load(curr_path, map_location="cpu"))

                    all_predictions = []
                    all_ys = []
                    for graphs, y in curr_testing_dl:
                        with torch.no_grad():
                            if graphs is None:
                                all_predictions.append(0.)
                                all_ys.append(y.float().detach().cpu())
                                continue

                            graphs, y = (graphs.to(device), y.to(device))

                            predictions = curr_classifier(graphs)
                            all_predictions.append(predictions.flatten().detach().cpu())
                            all_ys.append(y.float().detach().cpu())

                    predictions = np.concatenate(all_predictions)
                    y = np.concatenate(all_ys)
                    pdb_ids = [pdb_id for _ in range(len(y))]

                    curr_row = f"{pdb_id},{seed},{epoch},"
                    curr_row += get_performance(y, predictions, pdb_ids)
                    rows.append(curr_row)
        except Exception as e:
            print(e, pdb_id, decoy_style, direction)

    with open(f"results/{decoy_style}_{direction}.csv", "w") as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=0)
    args = vars(parser.parse_args())

    for decoy_style in ("ZS", "DS"):
        for direction in ("btd", "dtb", "complete"):
            write_eval(decoy_style, direction, args)

