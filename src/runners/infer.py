from ..model import SentimentClassifier
from transformers import BertTokenizer
from ..dataset.sentiment_dataloader import labels_to_idx

def infer(text, ckpt, pretrained_model_name = "bert-base-cased", max_len=64):
    model = SentimentClassifier.load_from_checkpoint(ckpt)
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
