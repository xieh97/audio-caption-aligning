## ICASSP 2022 - Unsupervised Audio-Caption Aligning Learns Correspondences between Individual Sound Events and Textual Phrases

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

## Code Tutorial

The codebase is developed with Python 3.8.

1. Checkout the code and install the required python packages:

```bash
git clone https://github.com/xieh97/audio-caption-alignment.git
pip install -r requirements.txt
```

2. Download audio clips and metadata (captions, event phrases, etc.) from the [AudioGrounding](https://github.com/wsntxxn/TextToAudioGrounding) dataset.

```
AudioGrounding
├─test.json
├─train.json
├─val.json
├─test
│  └─audio
│      └─...(70 wavs)
├─train
│  └─audio
│      └─...(4489 wavs)
└─val
    └─audio
        └─...(31 wavs)
```

3. Preprocess data:

```bash
python3 scripts/preprocess.py
```

4. Extract audio features:

```bash
python3 scripts/audio_features.py
```

5. Generate word embeddings:

```bash
python3 scripts/word_embeddings.py
```

6. Modify ``conf.yaml`` and train the model:

```bash
python3 main.py
```

7. Modify ``analysis/analysis_conf.yaml`` and evaluate the trained model on audio-caption retrieval and phrase-based sound event detection (SED):

```bash
python3 analysis/eval_checkpoint.py
```

8. Visualize samples:

```bash
python3 analysis/visualize_samples.py
```