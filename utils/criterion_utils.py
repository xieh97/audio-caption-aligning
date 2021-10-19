import numpy as np
import torch
import torch.nn as nn


class TripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0, reduction="STSQ"):
        super().__init__()

        self.margin = margin
        self.reduction = reduction

    def forward(self, audio_embeds, query_embeds, labels, audio_lens, query_lens, infos):
        """
        :param audio_embeds: tensor, (N, T, E).
        :param query_embeds: tensor, (N, Q, E).
        :param labels: tensor, (N, T).
        :param audio_lens: numpy 1D-array, (N,).
        :param query_lens: numpy 1D-array, (N,).
        :param infos: list of audio infos.
        :return:
        """
        N = audio_embeds.size(0)

        # Computes the triplet margin ranking loss for each anchor audio/query pair.
        # The impostor audio/query is randomly sampled from the mini-batch.
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        for i in range(N):
            A_imp_idx = i
            while infos[A_imp_idx]["ytid"] == infos[i]["ytid"]:
                A_imp_idx = np.random.randint(0, N)

            Q_imp_idx = i
            while infos[Q_imp_idx]["ytid"] == infos[i]["ytid"]:
                Q_imp_idx = np.random.randint(0, N)

            anchor_score = score(audio_embeds[i, 0:audio_lens[i]],
                                 query_embeds[i, 0:query_lens[i]],
                                 reduction=self.reduction)

            A_imp_score = score(audio_embeds[A_imp_idx, 0:audio_lens[A_imp_idx]],
                                query_embeds[i, 0:query_lens[i]],
                                reduction=self.reduction)

            Q_imp_score = score(audio_embeds[i, 0:audio_lens[i]],
                                query_embeds[Q_imp_idx, 0:query_lens[Q_imp_idx]],
                                reduction=self.reduction)

            A2Q_diff = self.margin + Q_imp_score - anchor_score
            if (A2Q_diff.data > 0.).all():
                loss = loss + A2Q_diff

            Q2A_diff = self.margin + A_imp_score - anchor_score
            if (Q2A_diff.data > 0.).all():
                loss = loss + Q2A_diff

        loss = loss / N

        return loss


def compute_similarity(audio_embed, query_embed):
    """
    Compute an audio-text alignment matrix.

    :param audio_embed: tensor, (T, E).
    :param query_embed: tensor, (Q, E).
    :return: similarity matrix: tensor, (T, Q).
    """

    # Compute dot products
    sim = torch.mm(audio_embed, query_embed.t())  # [T, Q]

    return torch.clamp(sim, min=0.)


def compute_similarities(audio_embeds, query_embeds, audio_lens):
    """
    Compute a batch of audio-text alignment matrices.

    :param audio_embeds: tensor, (N, T, E).
    :param query_embeds: tensor, (N, Q, E).
    :param audio_lens: numpy 1D-array, (N,).
    :return: tensor, (N, T, Q).
    """
    N = audio_embeds.size(0)

    match_maps = torch.zeros((N, audio_embeds.size(1), query_embeds.size(1)),
                             device=audio_embeds.device, requires_grad=False)
    for i in range(N):
        sim = compute_similarity(audio_embeds[i, 0:audio_lens[i]], query_embeds[i])  # [T, Q]
        match_maps[i, 0:audio_lens[i]] = sim

    return match_maps


def score(audio_embed, query_embed, reduction):
    """
    Calculate an audio-text similarity score.

    :param audio_embed: tensor, (T, E).
    :param query_embed: tensor, (Q, E).
    :param reduction: str, default "STSQ".
    :return: tensor, (1,).
    """

    sim = compute_similarity(audio_embed, query_embed)  # [T, Q]

    if reduction == "STSQ":
        return sim.mean()

    elif reduction == "MTSQ":
        return sim.max(dim=0).values.mean()

    elif reduction == "MQST":
        return sim.max(dim=1).values.mean()
