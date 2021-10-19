import torch
import torch.nn as nn


class WordEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, weight=None, trainable=True):
        super(WordEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if weight is not None:
            self.load_pretrained_embedding(weight, trainable)
        else:
            nn.init.kaiming_uniform_(self.embedding.weight)

    def load_pretrained_embedding(self, weight, trainable=True):
        assert weight.shape[0] == self.embedding.weight.size()[0], "vocabulary size mismatch!"

        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)

        for para in self.embedding.parameters():
            para.requires_grad = trainable

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, query_max_len, embed_dim).
        """

        query_lens = torch.as_tensor(query_lens)
        batch_size, query_max = queries.size()

        query_embeds = self.embedding(queries)

        mask = torch.arange(query_max, device="cpu").repeat(batch_size).view(batch_size, query_max)
        mask = (mask < query_lens.view(-1, 1)).to(query_embeds.device)

        query_embeds = query_embeds * mask.unsqueeze(-1)

        return query_embeds
