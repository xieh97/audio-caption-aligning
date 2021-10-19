import glob
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mutagen.wave import WAVE

from utils import word_utils

global_params = {
    "dataset_dir": "~/AudioGrounding",
    "clip_dir": "audio",
    "data_splits": ["train", "val", "test"],
    "jsons": ["train.json", "val.json", "test.json"],
}

#
# %% 1. Check available clips
#

clip_ytids = {}
clip_durations = {}

for data_split in global_params["data_splits"]:
    ytids, durations = [], []
    for clip_path in glob.glob(r"{}/*.wav".format(
            os.path.join(global_params["dataset_dir"], data_split, global_params["clip_dir"]))):
        try:
            clip = WAVE(clip_path)

            if clip.info.length > 0.0:
                ytid = re.sub(r"(_[0-9.]+){2}(.wav)", "", os.path.basename(clip_path))

                ytids.append(ytid)
                durations.append(clip.info.length)
        except:
            print("Error file: {}.".format(clip_path))
    clip_ytids[data_split] = ytids
    clip_durations[data_split] = durations

# Save clip info
with open(os.path.join(global_params["dataset_dir"], "clip_info.pkl"), "wb") as store:
    pickle.dump({"clip_ytids": clip_ytids, "clip_durations": clip_durations}, store)
print("Saved clip info")

#
# %% 2. Load and preprocess json records
#

phrase_dfs = {}
caption_dfs = {}

for data_split, json_file in zip(global_params["data_splits"], global_params["jsons"]):
    df = pd.read_json(os.path.join(global_params["dataset_dir"], json_file))

    df["filename"] = df["audio_id"]
    df["caption"] = df["tokens"]

    df[["ytid", "ts_start", "ts_end"]] = df.filename.str.extract(r"([-\w]+)_(\d+\.\d+)_(\d+\.\d+).wav")
    df[["ts_start", "ts_end"]] = df[["ts_start", "ts_end"]].apply(pd.to_numeric)

    df = df.loc[df["ytid"].isin(clip_ytids[data_split])]

    # Save clip-phrase pairs
    df_phrases = df[["audiocap_id", "filename", "soundtag", "timestamps"]].copy(deep=True)

    phrase_original, phrases, phrase_tokens = [], [], []
    for text in df_phrases["soundtag"]:
        phrase_original.append(text)

        text, words = word_utils.clean_text(text)
        phrases.append(text)
        phrase_tokens.append(words)

    df_phrases["original"] = phrase_original
    df_phrases["phrase"] = phrases
    df_phrases["phrase_tokens"] = phrase_tokens

    df_phrases = df_phrases[["audiocap_id", "filename", "original", "phrase", "phrase_tokens", "timestamps"]]
    df_phrases.to_json(os.path.join(global_params["dataset_dir"], "{}_{}.json".format(data_split, "phrase_pairs")))
    print("Saved", "{}_{}.json".format(data_split, "phrase_pairs"))

    phrase_dfs[data_split] = df_phrases

    # Save clip-caption pairs
    df_captions = df.groupby(by=["audiocap_id", "filename", "caption"])["timestamps"].apply(
        lambda x: np.concatenate(list(x))).reset_index()
    df_captions["timestamps"] = df_captions["timestamps"].apply(lambda x: list(set([tuple(i) for i in x])))

    caption_original, captions, caption_tokens = [], [], []
    for text in df_captions["caption"]:
        caption_original.append(text)

        text, words = word_utils.clean_text(text)
        captions.append(text)
        caption_tokens.append(words)

    df_captions["original"] = caption_original
    df_captions["caption"] = captions
    df_captions["caption_tokens"] = caption_tokens

    df_captions = df_captions[["audiocap_id", "filename", "original", "caption", "caption_tokens", "timestamps"]]
    df_captions.to_json(os.path.join(global_params["dataset_dir"], "{}_{}.json".format(data_split, "caption_pairs")))
    print("Saved", "{}_{}.json".format(data_split, "caption_pairs"))

    caption_dfs[data_split] = df_captions

#
# %% 3. Gather data statistics
#

# 1) clips
# 2) captions
# 3) event phrases
# 4) word frequencies

vocabulary = set()
word_bags = {}
split_infos = {}

for data_split in global_params["data_splits"]:
    ytids = clip_ytids[data_split]
    phrase_df = phrase_dfs[data_split]
    caption_df = caption_dfs[data_split]

    num_clips = caption_df["filename"].unique().size
    num_captions = caption_df.caption.size
    num_phrases = phrase_df.phrase.size

    assert len(ytids) == num_clips

    bag = []
    for tokens in caption_df["caption_tokens"]:
        bag.extend(tokens)
        vocabulary = vocabulary.union(tokens)

    num_words = len(bag)
    word_bags[data_split] = bag
    split_infos[data_split] = {
        "num_clips": num_clips,
        "num_captions": num_captions,
        "num_phrases": num_phrases,
        "num_words": num_words
    }

# Save vocabulary
with open(os.path.join(global_params["dataset_dir"], "vocab_info.pkl"), "wb") as store:
    pickle.dump({
        "vocabulary": vocabulary,
        "word_bags": word_bags,
        "split_infos": split_infos
    }, store)
print("Saved vocabulary info")


#
# %% 4. Plots
#

def hist_clip_duration(title, duration):
    fig = plt.figure(figsize=(6, 4))

    n, bins, _ = plt.hist(duration, bins=10, range=(0, 10), density=None)
    plt.xticks(ticks=bins, labels=bins)
    plt.xlabel("Clip duration (s)")
    plt.ylabel("Number of clips")
    plt.title(title)
    plt.show()

    fig.savefig(fname=os.path.join(global_params["dataset_dir"], "{}_clip_durations.svg".format(title)),
                format="svg", bbox_inches="tight")


def hist_word_frequencies(title, words, n=100):
    fig = plt.figure(figsize=(16, 6))

    freq_dist = nltk.FreqDist(words)
    freq_dist.plot(n, title=title)

    fig.savefig(fname=os.path.join(global_params["dataset_dir"], "{}_words_top{}.svg".format(title, n)),
                format="svg", bbox_inches="tight")


for data_split in global_params["data_splits"]:
    hist_clip_duration(data_split, clip_durations[data_split])
    hist_word_frequencies(data_split, word_bags[data_split])
