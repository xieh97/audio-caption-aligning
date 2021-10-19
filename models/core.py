import torch.nn as nn
import torch.nn.functional as F

from models import audio_encoders, text_encoders


class CRNNWordModel(nn.Module):

    def __init__(self, input_dim, proj_dim, vocab_size, weight=None, trainable=True):
        super(CRNNWordModel, self).__init__()

        self.audio_encoder = audio_encoders.CRNNEncoder(input_dim, proj_dim)

        self.text_encoder = text_encoders.WordEncoder(vocab_size, proj_dim, weight, trainable)

    def forward(self, audio_feats, queries, query_lens):
        """
        :param audio_feats: tensor, (batch_size, time_steps, Mel_bands).
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, time_steps, embed_dim), (batch_size, query_max_len, embed_dim).
        """

        audio_embeds = self.audio_encoder(audio_feats)
        audio_embeds = F.normalize(audio_embeds, dim=-1)  # [N, T, E]

        query_embeds = self.text_encoder(queries, query_lens)
        query_embeds = F.normalize(query_embeds, dim=-1)  # [N, Q, E]

        # audio_embeds: [N, T, E]    query_embeds: [N, Q, E]
        return audio_embeds, query_embeds
