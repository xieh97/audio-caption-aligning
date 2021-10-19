import os

import torch
import yaml

from utils import data_utils, model_utils, criterion_utils

# Load configurations
with open("analysis/analysis_conf.yaml", "rb") as stream:
    conf = yaml.safe_load(stream)

# Load audio-caption and audio-phrase pairs
caption_datasets, vocabulary = data_utils.load_data(conf["audio_caption"])
phrase_datasets, _ = data_utils.load_data(conf["audio_phrase"])

# Initialize a model instance
model_conf = conf[conf["model"]]
model = model_utils.get_model(model_conf, vocabulary)

# Load checkpoint states
checkpoint = conf["checkpoint"]
exp_path = os.path.join(conf["output_path"], conf["experiment"])
trial_path = os.path.join(exp_path, conf["trial"])

# Restore model states
model = model_utils.restore(model, os.path.join(trial_path, checkpoint))
model.eval()

# %%

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

colors = ["c", "r", "b", "g", "m", "y", "k"]
markers = [".", "*", "^", "s"]
linestyles = ["-", "--"]


def visualize_sample(sample_info):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))

    # Plot audio features
    xlim, ylim = sample_info["feature"].shape
    axs[0].imshow(sample_info["feature"].transpose(), aspect="auto", interpolation="bilinear", alpha=0.6,
                  origin="lower", extent=(-0.5, xlim + 0.5, -0.5, ylim + 0.5))
    axs[0].set_ylabel("# Mel", fontsize=16)
    axs[0].set_title("Log mel spectrogram")

    # Plot frame-phrase alignment similarities
    for c_ind, event in enumerate(sample_info["events"], 1):
        axs[1].plot(event["mean"], label=event["phrase"], color=colors[c_ind], linestyle="-")

        for x1, x2 in find_regions(event["label"]):
            axs[0].add_patch(plt.Rectangle(xy=(x1, 0), width=x2 - x1, height=ylim, color=colors[c_ind],
                                           fill=False, linewidth=(2.5 - 0.5 * c_ind)))

    axs[1].grid()
    # axs[1].legend(loc="upper right")
    axs[1].legend()
    axs[1].set_xlabel("Time (s)", fontsize=16)
    axs[1].set_ylabel("Similarity", fontsize=16)

    plt.suptitle("Caption: {0}".format(sample_info["caption"]), fontsize=18)
    plt.tight_layout()

    fname = "{0}_{1}".format(sample_info["ytid"], sample_info["audiocap_id"])

    return fig, fname


def find_regions(label):
    indexes = np.logical_xor(label[1:], label[:-1]).nonzero()[0]
    indexes += 1

    if label[0]:
        indexes = np.r_[0, indexes]

    if label[-1]:
        indexes = np.r_[indexes, label.size]

    indexes = indexes.reshape((-1, 2))
    return indexes


# %%


def compute_similarity(model, audio_feat, txt_query, reduction):
    audio_feat = torch.unsqueeze(audio_feat, dim=0)
    txt_query = torch.unsqueeze(txt_query, dim=0)

    feat_embs, query_embs = model(audio_feat, txt_query, [txt_query.size()[-1]])

    alignment_matrix = criterion_utils.compute_similarity(feat_embs[0], query_embs[0])

    if reduction == "mean":
        alignment_matrix = alignment_matrix.mean(dim=1, keepdim=False)
    elif reduction == "max":
        alignment_matrix = alignment_matrix.max(dim=1, keepdim=False).values

    # Min-max normalization
    alignment_matrix -= alignment_matrix.min(dim=0, keepdim=True)[0]
    alignment_matrix /= alignment_matrix.max(dim=0, keepdim=True)[0]

    return alignment_matrix.detach()


# %%

# Visualize samples
for split in ["train", "val", "test"]:

    split_dir = os.path.join(trial_path, checkpoint, split)

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    cap_ds = caption_datasets[split]
    phr_ds = phrase_datasets[split]

    # Group event phrases according to their captions
    cap_phrases = {}
    for phr_ind in range(len(phr_ds)):
        phr_item = phr_ds.data_df.iloc[phr_ind]

        if cap_phrases.get(phr_item["audiocap_id"]) is None:
            cap_phrases[phr_item["audiocap_id"]] = [phr_ind]
        else:
            cap_phrases[phr_item["audiocap_id"]].append(phr_ind)

    # Calculate frame-phrase similarities
    for cap_ind in range(len(cap_ds)):
        audio_feat, cap_query, _, cap_info = cap_ds[cap_ind]

        phr_indexes = cap_phrases[cap_info["audiocap_id"]]

        if len(phr_indexes) < 3:

            event_infos = []

            for phr_ind in phr_indexes:
                _, phr_query, phr_label, phr_info = phr_ds[phr_ind]

                phr_sim = compute_similarity(model, audio_feat, phr_query, "mean")

                event_infos.append({
                    "index": phr_ind,
                    "phrase": phr_ds.data_df.iloc[phr_ind]["original"],
                    "label": phr_label.detach().numpy(),
                    "mean": phr_sim.numpy()
                })

            sample_info = {
                "ytid": cap_info["ytid"],
                "audiocap_id": cap_info["audiocap_id"],
                "feature": audio_feat.detach().numpy(),
                "caption": cap_ds.data_df.iloc[cap_ind]["original"],
                "tokens": cap_ds.data_df.iloc[cap_ind]["caption_tokens"],
                "events": event_infos
            }

            fig, fname = visualize_sample(sample_info)
            fig.savefig(fname=os.path.join(split_dir, "{}.png".format(fname)), format="png", bbox_inches="tight")
            plt.close(fig)
            print(fname)
