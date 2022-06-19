from ..model import SentimentClassifier
from transformers import BertTokenizer
from ..dataset.sentiment_dataloader import labels_to_idx

import os

loaded_ckpt = None
loaded_model = None


def infer(text, ckpt, pretrained_model_name="bert-base-cased", max_len=64):
    global loaded_model, loaded_ckpt
    print("loaded model is ", loaded_ckpt)
    if loaded_ckpt == None or os.path.normpath(loaded_ckpt) != os.path.normpath(ckpt):
        print("loading model", ckpt)
        model = SentimentClassifier.load_from_checkpoint(ckpt)
        loaded_ckpt = ckpt
        loaded_model = model
    else:
        model = loaded_model

    PRE_TRAINED_MODEL_NAME = pretrained_model_name
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    prob, pred = model.infer(text,
                             tokenizer=tokenizer,
                             max_len=max_len)
    pred = pred.cpu().item()
    for k, v in list(labels_to_idx.items()):
        if pred == v:
            pred = k

    return prob.cpu().item(), pred
