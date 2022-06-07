from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import sys

sys.path.append('src')

from ..model import SentimentClassifier
from ..dataset.sentiment_dataloader import Sentiment_Dataloader

# from utils.visualization_tools import PlotCMs


logger = logging.getLogger(__name__)


def _get_callbacks():
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        patience=5,
        min_delta=0,
        strict=True,
        verbose=False,
        mode="max",
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        dirpath=os.path.join("build", "models"),
        filename="Bert-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
        save_weights_only=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor()

    return [
        early_stop_callback,
        checkpoint_callback,
        lr_monitor,
    ]


def get_dataloader(data_src_file, max_encoding_len,
                   batch_size=32, model_name='bert-base-cased'):
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_len = max_encoding_len

    loader = Sentiment_Dataloader(batch_size=batch_size,
                                  data_src_file=data_src_file,
                                  max_len=max_len, tokenizer=tokenizer)
    loader.setup()
    return loader, tokenizer


def trainer(conf, run_train=True, run_val=True, run_test=True, ckpt=None,
            gpu=1):
    logger.info("Staring training...")

    # conf = {
    #     "batch_size": 32,
    #     "model_name": "bert-base-cased",
    #     "max_input_len": 64,
    #     "data_src_file": "build/datasets/labelled_text.csv",
    #     "epochs": 10,
    #     "lr": 2e-5,
    #     "weight_decay": 1e-2,
    #     "dropout_rate": 0.5,
    # }

    epochs = conf["epochs"]
    max_len = conf["max_input_len"]
    PRE_TRAINED_MODEL_NAME = conf["model_name"]

    loader, tokenizer = get_dataloader(data_src_file=conf["data_src_file"],
                   max_encoding_len=max_len,
                   batch_size=conf["batch_size"],
                   model_name=PRE_TRAINED_MODEL_NAME)

    callbacks = _get_callbacks()
    model_conf = {"total_steps": len(loader.train_dataset) * epochs,
                  "lr": conf["lr"],
                  "weight_decay": conf["weight_decay"],
                  "dropout_rate": conf["dropout_rate"],
                  "pre_trained_model_name": conf["model_name"],
                  }
    if ckpt is None:
        model = SentimentClassifier(model_conf)
    elif ckpt is not None:
        model = SentimentClassifier.load_from_checkpoint(ckpt)

    stats_logger = None
    if run_train:
        stats_logger = TensorBoardLogger(
            os.path.join("build", "tf_logs"),
            name="Bert",
            flush_secs=30,
        )

    trainer = pl.Trainer(check_val_every_n_epoch=1, fast_dev_run=False, max_epochs=epochs,
                         gpus=gpu, logger=stats_logger, callbacks=callbacks, gradient_clip_val=1)

    if run_train:
        trainer.fit(model, loader)

    # plotter = PlotCMs()
    if run_val:
        logger.info("Validating...")
        trainer.test(model, loader.val_dataloader())
        cm_value = np.round(model.cm.compute().cpu().numpy(), 3)
        print("Validation stats", str(cm_value))
        # plotter.insert(cm_value, "Val CM")
        # model.cm.reset()

    if run_test:
        logger.info("Testing...")
        trainer.test(model, loader.test_dataloader())
        cm_value = np.round(model.cm.compute().cpu().numpy(), 3)
        print("Testing stats", str(cm_value))
        # plotter.insert(cm_value, "Test CM")

    ## todo: save plots
