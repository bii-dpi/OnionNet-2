from imports import *


def get_recall(precisions, recalls, prec_val):
    precisions = np.abs(np.array(precisions) - prec_val)

    return recalls[np.argmin(precisions)]


def get_ef(y, prec_val, num_pos):
    len_to_take = int(len(y) * prec_val)

    return (np.sum(y[:len_to_take]) / num_pos) * 100


def get_log_auc(predicted, y, num_pos):
    prec_vals = np.linspace(.0001, 1, 10000)
    recalls = []
    for prec_val in prec_vals:
        recalls.append(float(get_ef(y, prec_val, num_pos)))

    return np.trapz(y=recalls, x=np.log10(prec_vals) / 3, dx=1/30)


def get_performance(y, predictions, pdb_ids):
    y = np.array(y, dtype=int)

    prediction_dict = defaultdict(list)
    y_dict = defaultdict(list)
    for i in range(len(pdb_ids)):
        prediction_dict[pdb_ids[i]].append(predictions[i])
        y_dict[pdb_ids[i]].append(y[i])

    all_aucs = []
    all_auprs = []
    all_mccs = []
    for pdb_id in prediction_dict:
        all_aucs.append(roc_auc_score(y_dict[pdb_id],
                                      prediction_dict[pdb_id]))
        all_auprs.append(average_precision_score(y_dict[pdb_id],
                                                 prediction_dict[pdb_id]))
        all_mccs.append(matthews_corrcoef(y_dict[pdb_id],
                                          np.round(prediction_dict[pdb_id])))

    auc = np.mean(all_aucs)
    aupr = np.mean(all_auprs)
    mcc = np.mean(all_mccs)

    all_laucs = []
    for pdb_id in prediction_dict:
        y = y_dict[pdb_id]
        y = np.array(y)
        predicted = prediction_dict[pdb_id]
        predicted = np.array(predicted)

        precisions, recalls, _ = \
            precision_recall_curve(y, predicted)

        sorted_indices = np.argsort(predicted)[::-1]
        y = y[sorted_indices]
        predicted = predicted[sorted_indices]
        num_pos = np.sum(y)

        curr_lauc = get_log_auc(predicted, y, num_pos)
        all_laucs.append(curr_lauc)
        #print(f"{pdb_id},{curr_lauc}")

    log_auc = np.mean(all_laucs)
    #print(all_laucs)

    results = [str(auc), str(aupr), str(log_auc), str(mcc)]

    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        all_recs = []
        for pdb_id in prediction_dict:
            y = y_dict[pdb_id]
            y = np.array(y)
            predicted = prediction_dict[pdb_id]
            predicted = np.array(predicted)

            precisions, recalls, _ = \
                precision_recall_curve(y, predicted)

            all_recs.append(get_recall(precisions, recalls, prec_val))

        results.append(str(np.mean(all_recs)))

    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        all_efs = []
        for pdb_id in prediction_dict:
            y = y_dict[pdb_id]
            y = np.array(y)
            predicted = prediction_dict[pdb_id]
            predicted = np.array(predicted)

            precisions, recalls, _ = \
                precision_recall_curve(y, predicted)

            sorted_indices = np.argsort(predicted)[::-1]
            y = y[sorted_indices]
            predicted = predicted[sorted_indices]
            num_pos = np.sum(y)

            all_efs.append(get_ef(y, prec_val, num_pos))

        results.append(str(np.mean(all_efs)))

    return ','.join(results)

