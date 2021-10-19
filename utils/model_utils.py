import os

import torch

from models import core


def get_model(config, vocabulary):
    model_args = config["args"]

    if config["name"] == "CRNNWordModel":
        audioEnc_args = model_args["audio_encoder"]
        textEnc_args = model_args["text_encoder"]

        return getattr(core, "CRNNWordModel", None)(
            input_dim=audioEnc_args["input_dim"],
            proj_dim=model_args["proj_dim"],
            vocab_size=len(vocabulary),
            weight=vocabulary.get_weights() if textEnc_args["pretrained"] else None,
            trainable=textEnc_args["trainable"]
        )

    return None


def train(model, optimizer, criterion, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.train()

    for batch_idx, data in enumerate(data_loader, 0):
        # Get the inputs; data is a list of tuples (audio_feats, audio_lens, queries, query_lens, labels, infos)
        audio_feats, audio_lens, queries, query_lens, labels, infos = data
        audio_feats, queries, labels = audio_feats.to(device), queries.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        audio_embeds, query_embeds = model(audio_feats, queries, query_lens)
        loss = criterion(audio_embeds, query_embeds, labels, audio_lens, query_lens, infos)
        loss.backward()
        optimizer.step()


def eval(model, criterion, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion.to(device=device)
    model.to(device=device)

    model.eval()

    eval_loss, eval_steps = 0.0, 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader, 0):
            audio_feats, audio_lens, queries, query_lens, labels, infos = data
            audio_feats, queries, labels = audio_feats.to(device), queries.to(device), labels.to(device)

            audio_embeds, query_embeds = model(audio_feats, queries, query_lens)

            loss = criterion(audio_embeds, query_embeds, labels, audio_lens, query_lens, infos)
            eval_loss += loss.cpu().numpy()
            eval_steps += 1

    return eval_loss / (eval_steps + 1e-20)


def restore(model, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"),
                                              map_location=device)
    model.load_state_dict(model_state)
    return model
