# Unsupervised Audio-Caption Aligning (ICASSP 2022)

This repository provides source code for [our paper](https://arxiv.org/abs/2110.02939):

```
@InProceedings{Xie2022Unsupervised,
    author = {Xie, Huang and Räsänen, Okko and Drossos, Konstantinos and Virtanen, Tuomas},
    title = {Unsupervised Audio-Caption Aligning Learns Correspondences Between Individual Sound Events and Textual Phrases},
    booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = {2022},
    pages = {8867-8871},
    doi = {10.1109/ICASSP43922.2022.9747336}
}
```

# Code Tutorial

This codebase is developed with Python 3.8.

1. Check out source code and install required python packages:

```
git clone https://github.com/xieh97/audio-caption-alignment.git
pip install -r requirements.txt
```

2. Download audio clips and metadata (captions, event phrases, etc.) from
   the [AudioGrounding](https://github.com/wsntxxn/TextToAudioGrounding) dataset.

```
AudioGrounding
├─ test.json
├─ train.json
├─ val.json
├─ test
│  └─audio
│     └─...(70 wavs)
├─ train
│  └─audio
│     └─...(4489 wavs)
└─ val
   └─audio
      └─...(31 wavs)
```

3. Pre-process audio and caption data:

```
scripts
├─ preprocess.py              # clean audio captions
├─ audio_features.py          # extract log-mel energies from audio clips
└─ word_embeddings.py         # generate word embeddings using pretrained Word2Vec
```

4. Train the audio-text aligning system ([More details](https://arxiv.org/abs/2110.02939)):

```
models
├─ core.py                    # dual-encoder framework
├─ audio_encoders.py          # audio encoders
└─ text_encoders.py           # text encoders

utils
├─ criterion_utils.py         # loss functions
├─ data_utils.py              # Pytorch dataset classes
├─ metric_utils.py            # SED metrics, retrieval metrics
├─ model_utils.py             # model.train(), model.eval(), etc.
└─ word_utils.py              # pretrained Word2Vec

conf.yaml                     # experimental settings
main.py                       # main()
```

5. Evaluate the trained model on audio-caption retrieval and phrase-based
   sound event detection (SED):

```
analysis
├─ analysis_conf.yaml         # model checkpoint
├─ eval_checkpoint.py         # calculate metrics for SED and retrieval
└─ visualize_samples.py       # visualize audio-text aligning
```
