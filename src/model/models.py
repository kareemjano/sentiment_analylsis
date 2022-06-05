from torch import nn
from transformers import BertModel
from .base import ClassificationModule
from torchmetrics import ConfusionMatrix


class SentimentClassifier(ClassificationModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_classes = 4
        self.model = SentimentModel(self.n_classes,
                                    pre_trained_model_name=self.pre_trained_model_name,
                                    dropout=self.dropout_rate)
        self.cm = ConfusionMatrix(num_classes=self.n_classes,
                                  normalize="true")

class SentimentModel(nn.Module):

    def __init__(self, n_classes, pre_trained_model_name, dropout=0.3):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        output = self.drop(pooled_output)
        return self.out(output)