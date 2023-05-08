from imports import *
from model import Classifier
from dataloader import get_training_dataloader


LR = 1e-3
BATCH_SIZE = 10
EPOCHS = 50
#EPOCHS = 4

def get_bce_loss(predictions, y):
    BCE = F.binary_cross_entropy(predictions.flatten(), y.float(), reduction="sum")

    return BCE


def get_curr_epoch(decoy_style, direction, seed):
    model_fnames = [fname.split("_")[-1].replace(".pt", "") for fname in
                    os.listdir(f"models/{decoy_style}/{direction}/")
                    if fname.startswith(str(seed))]
    if not model_fnames:
        return 0

    curr_epoch = sorted([int(epoch) for epoch in model_fnames])[-1]

    return curr_epoch


def save_trained(decoy_style, direction, seed):

    os.system(f"mkdir models/{decoy_style}/ 2>/dev/null")
    os.system(f"mkdir models/{decoy_style}/{direction} 2>/dev/null")

    device = torch.device(f"cuda:{args['c']}")
    training_dl = get_training_dataloader(decoy_style, direction, seed, BATCH_SIZE)

    classifier = Classifier().to(device)

    curr_epoch = get_curr_epoch(decoy_style, direction, seed)
    if args["r"]:
        curr_epoch = 0

    if curr_epoch:
        fname = f"models/{decoy_style}/{direction}/{seed}_{curr_epoch}.pt"
        classifier.load_state_dict(torch.load(fname, map_location="cpu"))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    start_from = curr_epoch + 1 if curr_epoch else 0

    for epoch in progressbar(range(start_from, EPOCHS)):
        for iteration, (graphs, y) in enumerate(training_dl):
            optimizer.zero_grad()
            graphs, y = (graphs.to(device), y.to(device))

            predictions = classifier(graphs)
            bce = get_bce_loss(predictions, y)
            bce.backward()

            optimizer.step()

        torch.save(classifier.state_dict(),
                   f"models/{decoy_style}/{direction}/{seed}_{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=0)
    parser.add_argument('-r', type=int, default=0)
    args = vars(parser.parse_args())

    for decoy_style in ("ZS", "DS"):
        for direction in ("btd", "dtb", "complete"):
            for seed in (range(NUM_SEEDS)):
                print(decoy_style, direction, seed)
                save_trained(decoy_style, direction, seed)

