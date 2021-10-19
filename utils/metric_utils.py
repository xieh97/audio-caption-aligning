import numpy as np
import torch
import torchmetrics

from utils import criterion_utils


def frame_based_metrics(model, data_loader, threshold, reduction):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    model.eval()

    precision = torchmetrics.Precision(threshold=threshold, average="micro", compute_on_step=False)
    precision.to(device=device)

    recall = torchmetrics.Recall(threshold=threshold, average="micro", compute_on_step=False)
    recall.to(device=device)

    f1 = torchmetrics.F1(threshold=threshold, average="micro", compute_on_step=False)
    f1.to(device=device)

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            audio_feats, audio_lens, queries, query_lens, labels, infos = data
            audio_feats, queries, labels = audio_feats.to(device), queries.to(device), labels.to(device)

            if reduction == "baseline":
                alignment_matrices = torch.rand_like(labels, device=device)

            else:
                audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

                # Alignment matrices [N, T, Q]
                alignment_matrices = criterion_utils.compute_similarities(audio_embeds, query_embeds, audio_lens)

                # Aggregate along Q
                if reduction == "mean":
                    alignment_matrices = alignment_matrices.mean(dim=2, keepdim=False)  # [N, T]
                elif reduction == "max":
                    alignment_matrices = alignment_matrices.max(dim=2, keepdim=False).values  # [N, T]

            # Min-max normalization
            alignment_matrices -= alignment_matrices.min(dim=1, keepdim=True)[0]
            alignment_matrices /= alignment_matrices.max(dim=1, keepdim=True)[0]

            # Frame-based metric
            precision(alignment_matrices, labels.long())
            recall(alignment_matrices, labels.long())
            f1(alignment_matrices, labels.long())

        precision = precision.compute().item()
        recall = recall.compute().item()
        f1 = f1.compute().item()

    return {"precision": precision, "recall": recall, "f1": f1}


def retrieval_metrics(model, dataset, reduction):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    model.eval()

    # Initial metric instances
    recalls = {"a2c_R": {}, "c2a_R": {}}
    for task in recalls:
        for k in [1, 5]:
            instance = torchmetrics.RetrievalRecall(empty_target_action="skip", compute_on_step=False, k=k)
            instance.to(device=device)

            recalls[task]["{}{}".format(task, k)] = instance

    with torch.no_grad():
        # Group audio-caption pairs by ytid
        ytid_group = {}

        for i in range(len(dataset)):
            item = dataset.data_df.iloc[i]

            if ytid_group.get(item["ytid"]) is None:
                ytid_group[item["ytid"]] = [i]
            else:
                ytid_group[item["ytid"]].append(i)

        # ytid-to-index, index-to-ytid
        ytid2ind = {ytid: ind for ind, ytid in enumerate(ytid_group, 0)}
        ind2ytid = {ytid2ind[ytid]: ytid for ytid in ytid2ind}

        # Randomize 30 audio-caption pairs (1 ground-truth + 29 non-positive) for each audio sample (ytid)
        for i_ytid in ytid_group:
            indexes, a2c_preds, c2a_preds, target = [], [], [], []

            # Select the ground truth
            i = np.random.choice(a=ytid_group[i_ytid], size=1, replace=False, p=None)[0]
            i_audio_emb, i_query_emb, i_info = transform(model, dataset, i, device)

            gt_score = criterion_utils.score(i_audio_emb, i_query_emb, reduction=reduction)

            indexes.append(i)
            a2c_preds.append(gt_score.item())
            c2a_preds.append(gt_score.item())
            target.append(i_info["ytid"] == i_ytid)

            # Select 29 non-positive audio samples (ytids)
            num_items = 30
            ytid_indexes = np.array([ind for ind in ind2ytid])
            if len(ytid_indexes) > num_items:
                probs = np.array([ytid2ind[i_ytid] != ind for ind in ytid_indexes])
                probs = probs / (len(ytid_indexes) - 1)

                ytid_indexes[:num_items - 1] = np.random.choice(a=ytid_indexes, size=num_items - 1, replace=False,
                                                                p=probs)
                ytid_indexes[num_items - 1] = ytid2ind[i_ytid]
                ytid_indexes = ytid_indexes[:num_items]

                assert len(ytid_indexes) == num_items
                assert ytid_indexes[num_items - 1] == ytid2ind[i_ytid]

            # Randomize 29 non-positives
            for ind in ytid_indexes[:num_items - 1]:
                j_ytid = ind2ytid[ind]

                j = np.random.choice(a=ytid_group[j_ytid], size=1, replace=False, p=None)[0]

                j_audio_emb, j_query_emb, j_info = transform(model, dataset, j, device)

                a2c_score = criterion_utils.score(i_audio_emb, j_query_emb, reduction=reduction)
                c2a_score = criterion_utils.score(j_audio_emb, i_query_emb, reduction=reduction)

                indexes.append(i)
                a2c_preds.append(a2c_score.item())
                c2a_preds.append(c2a_score.item())
                target.append(i_info["ytid"] == j_info["ytid"])

            indexes = torch.tensor(indexes, device=device, dtype=torch.long)
            a2c_preds = torch.tensor(a2c_preds, device=device)
            c2a_preds = torch.tensor(c2a_preds, device=device)
            target = torch.tensor(target, device=device)

            # Update metrics
            for key in recalls["a2c_R"]:
                instance = recalls["a2c_R"][key]
                instance(a2c_preds, target, indexes=indexes)

            for key in recalls["c2a_R"]:
                instance = recalls["c2a_R"][key]
                instance(c2a_preds, target, indexes=indexes)

        # Compute metrics
        for task in recalls:
            for key in recalls[task]:
                instance = recalls[task][key]
                recalls[task][key] = instance.compute().item()

    return recalls


def transform(model, dataset, index, device=None):
    audio, query, _, info = dataset[index]

    audio = torch.unsqueeze(audio, dim=0).to(device=device)
    query = torch.unsqueeze(query, dim=0).to(device=device)

    audio_emb, query_emb = model(audio, query, [query.size(-1)])

    audio_emb = torch.squeeze(audio_emb, dim=0).to(device=device)
    query_emb = torch.squeeze(query_emb, dim=0).to(device=device)

    return audio_emb, query_emb, info
