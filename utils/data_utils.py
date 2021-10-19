import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Vocabulary(object):

    def __init__(self):
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.weights = None

    def add_word(self, word, word_vector):
        if word not in self.word2idx:
            self.word2vec[word] = word_vector
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_weights(self):
        for idx in range(self.idx):
            if self.weights is None:
                self.weights = self.word2vec[self.idx2word[idx]]
            else:
                self.weights = np.vstack((self.weights, self.word2vec[self.idx2word[idx]]))

        return self.weights

    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class QueryAudioDataset(Dataset):

    def __init__(self, audio_feature, data_df, query_col, vocabulary, win_shift=0.02):
        self.audio_feature = audio_feature
        self.data_df = data_df
        self.query_col = query_col
        self.vocabulary = vocabulary
        self.win_shift = win_shift

    def __getitem__(self, index):
        item = self.data_df.iloc[index]

        audio_feat = torch.as_tensor(self.audio_feature[item["ytid"]][()])
        query = torch.as_tensor([self.vocabulary(token) for token in item[self.query_col]])

        label = np.zeros(audio_feat.shape[0], dtype=int)
        for start, end in item["timestamps"]:
            start = np.ceil(start / self.win_shift).astype(int)
            end = np.ceil(end / self.win_shift).astype(int)
            label[start: end] = 1
        label = torch.as_tensor(label)

        info = {"audiocap_id": item["audiocap_id"], "ytid": item["ytid"],
                "start_word": None, "timestamps": item["timestamps"]}

        return audio_feat, query, label, info

    def __len__(self):
        return len(self.data_df)


def collate_fn(data_batch):
    """
    :param data_batch: a list of tensor tuples (audio_feat, query, label, info).
    :return:
    """
    audio_feat_batch = []
    query_batch = []
    label_batch = []
    info_batch = []

    for a, q, l, i in data_batch:
        audio_feat_batch.append(a)
        query_batch.append(q)
        label_batch.append(l)
        info_batch.append(i)

    audio_feat_batch, audio_feat_lens = pad_tensors(audio_feat_batch)
    query_batch, query_lens = pad_tensors(query_batch)
    label_batch, _ = pad_tensors(label_batch)

    return audio_feat_batch.float(), audio_feat_lens, query_batch.long(), query_lens, label_batch.float(), info_batch


def pad_tensors(tensor_list):
    tensor_lens = [tensor.shape for tensor in tensor_list]

    dim_max_lens = tuple(np.max(tensor_lens, axis=0))

    tensor_lens = np.array(tensor_lens)[:, 0]

    padded_tensor = torch.zeros((len(tensor_list),) + dim_max_lens)
    for i, t in enumerate(tensor_list):
        end = tensor_lens[i]
        padded_tensor[i, :end] = t[:end]

    return padded_tensor, tensor_lens


def load_data(config):
    # Load audio features
    audio_feats = h5py.File(os.path.join(config["input_path"], config["audio_features"]), "r")

    # Load pretrained word embeddings
    with open(os.path.join(config["input_path"], config["word_embeddings"]), "rb") as emb_reader:
        word_vectors = pickle.load(emb_reader)

    # Construct vocabulary
    vocabulary = Vocabulary()
    for word in word_vectors:
        if len(vocabulary) == 0:
            vocabulary.add_word("<pad>", np.zeros_like(word_vectors[word]))
        vocabulary.add_word(word, word_vectors[word])

    # Load data splits
    text_datasets = {}
    for split in ["train", "val", "test"]:
        df = pd.read_json(os.path.join(config["input_path"], config["data_splits"][split]))
        df[["ytid", "ts_start", "ts_end"]] = df.filename.str.extract(r"([-\w]+)_(\d+\.\d+)_(\d+\.\d+).wav")
        dataset = QueryAudioDataset(audio_feats, df, config["text_tokens"], vocabulary)
        text_datasets[split] = dataset

    return text_datasets, vocabulary
