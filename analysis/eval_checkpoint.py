import os
import pickle

import numpy as np
import yaml
from torch.utils.data import DataLoader

from utils import data_utils, model_utils, criterion_utils, metric_utils

# Load configurations
with open("analysis/analysis_conf.yaml", "rb") as stream:
    conf = yaml.safe_load(stream)

# Load audio-caption data
caption_datasets, vocabulary = data_utils.load_data(conf["audio_caption"])
caption_loaders = {}
for split in ["train", "val", "test"]:
    caption_loaders[split] = DataLoader(dataset=caption_datasets[split],
                                        batch_size=32, shuffle=False, collate_fn=data_utils.collate_fn)

# Load audio-phrase data
phrase_datasets, _ = data_utils.load_data(conf["audio_phrase"])
phrase_loaders = {}
for split in ["train", "val", "test"]:
    phrase_loaders[split] = DataLoader(dataset=phrase_datasets[split],
                                       batch_size=1, shuffle=False, collate_fn=data_utils.collate_fn)

# Initialize a loss object
criterion_conf = conf[conf["criterion"]]
criterion = getattr(criterion_utils, criterion_conf["name"], None)(**criterion_conf["args"])

# Initialize a model object
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

# Calculate audio-caption retrieval metrics: recall at {1, 5}
retrieval_metrics = {}
for split in ["test"]:

    split_metrics = {
        "a2c_R1": [], "a2c_R5": [],
        "c2a_R1": [], "c2a_R5": []
    }
    retrieval_metrics[split] = split_metrics

    for i in range(20):
        metrics = metric_utils.retrieval_metrics(model, caption_datasets[split],
                                                 reduction=criterion_conf["args"]["reduction"])

        for task in metrics:
            for key in metrics[task]:
                split_metrics[key].append(metrics[task][key])

# Print audio-caption retrieval metrics: recall at {1, 5}
for split in retrieval_metrics:
    for key in retrieval_metrics[split]:
        print(split, key, np.mean(retrieval_metrics[split][key]), np.std(retrieval_metrics[split][key]))

# Save retrieval metrics
with open(os.path.join(trial_path, checkpoint, "retrieval_metrics.pkl"), "wb") as store:
    pickle.dump({
        "retrieval_metrics": retrieval_metrics
    }, store)
print("Save", "retrieval_metrics", "retrieval_metrics.pkl")

# %%

# Calculate phrase-based SED metrics: precision, recall, f1
thresholds = np.arange(0.01, 1.00, 0.01)
sed_metrics = {}
for split in ["test"]:

    split_metrics = {
        "baseline": {"precision": [], "recall": [], "f1": []},
        "mean": {"precision": [], "recall": [], "f1": []},
        "max": {"precision": [], "recall": [], "f1": []}
    }
    sed_metrics[split] = split_metrics

    for threshold in thresholds:
        for reduction in split_metrics:
            metrics = metric_utils.frame_based_metrics(model, phrase_loaders[split], threshold, reduction=reduction)

            for key in metrics:
                split_metrics[reduction][key].append(metrics[key])

# Save retrieval metrics
with open(os.path.join(trial_path, checkpoint, "sed_metrics.pkl"), "wb") as store:
    pickle.dump({
        "sed_metrics": sed_metrics
    }, store)
print("Save", "sed_metrics", "sed_metrics.pkl")

# %%

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
colors = ["blue", "red", "green"]

fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={"aspect": "auto", "xlim": (0., 1.), "ylim": (0., 1.)},
                        figsize=(9, 3))

for split in sed_metrics:
    split_metrics = sed_metrics[split]

    for cidx, reduction in enumerate(split_metrics):
        for ncol, metric in enumerate(split_metrics[reduction]):
            axs[ncol].plot(thresholds, split_metrics[reduction][metric], label=reduction, color=colors[cidx])

            if cidx == len(split_metrics) - 1:
                axs[ncol].grid()
                axs[ncol].legend()
                axs[ncol].set_xlabel("Threshold", fontsize=12)
                axs[ncol].set_ylabel(metric.capitalize(), fontsize=12)

    plt.tight_layout()
    fig.savefig(fname=os.path.join(trial_path, checkpoint, "{}_sed_metrics.png".format(split)), format="png",
                bbox_inches="tight")
    fig.savefig(fname=os.path.join(trial_path, checkpoint, "{}_sed_metrics.svg".format(split)), format="svg",
                bbox_inches="tight")
