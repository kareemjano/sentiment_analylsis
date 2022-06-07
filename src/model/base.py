import torch
from torch import nn
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
from abc import ABC

class ClassificationModule(pl.LightningModule, ABC):
    def __init__(self, conf):
        """
        configuration parameters containing params and hparams.
        """
        super().__init__()
        self.save_hyperparameters()

        self.conf = conf
        self.total_steps = conf["total_steps"]
        self.lr = conf["lr"]
        self.weight_decay = conf["weight_decay"]
        self.dropout_rate = conf["dropout_rate"]
        self.pre_trained_model_name = conf["pre_trained_model_name"]

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.total_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        return [optimizer], [scheduler]

    def calc_acc(self, y, pred):
        assert pred.shape == y.shape, "shape of prediction doesnt match ground truth labels"
        acc = (pred == y).sum() / y.size(0)
        return acc

    def general_step(self, batch, mode):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = self.loss_fn(outputs, targets)

        n_correct = self.calc_acc(targets, preds)
        if mode == "test":
            self.cm(preds, targets)

        return loss, n_correct.cpu().detach()

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, n_correct = self.general_step(train_batch, "train")
        return {
            'loss': loss,
            'acc': n_correct,
        }

    def validation_step(self, val_batch, batch_idx, mode="val"):
        self.eval()
        loss, n_correct = self.general_step(val_batch, mode)
        return {
            'loss': loss.detach().cpu(),
            'acc': n_correct.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode="test")

    def general_epoch_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log(f'{mode}_loss', avg_loss, logger=logger, on_epoch=True)
        self.log(f'{mode}_acc', float(f"{avg_acc * 100:.2f}"), logger=logger, on_epoch=True)
        return avg_loss, avg_acc

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_acc = self.general_epoch_end(outputs, 'test')
        ret = {
            'avg_loss': avg_loss,
            'avg_acc': float(f"{avg_acc * 100:.2f}"),
        }
        print(ret)
        return ret

    def infer(self, text, tokenizer, max_len=64):
        self.eval()

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            output = torch.softmax(self(input_ids, attention_mask), dim=-1)
            prob, prediction = torch.max(output, dim=1)
        return prob, prediction