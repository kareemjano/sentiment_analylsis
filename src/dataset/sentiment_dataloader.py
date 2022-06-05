import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


labels_to_idx = {'Positive': 0, 'Neutral': 1, 'Negative': 2, 'Irrelevant': 3}


def load_data(path, column_names=None):
    df = pd.read_csv(path, encoding="ISO-8859-1", header=None)
    if column_names is not None:
        df.columns = column_names

    return df


def preprocess(df, mode="train"):
    df = df.dropna()
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    for i, chat in zip(df.index, df["chat"]):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        chat = re.sub(r'[\S]+\.(net|com|org|info|edu|gov|uk|de|ca|jp|fr|au|us|ru|ch|it|nel|se|no|es|mil)[\S]*\s?', '',
                      chat)
        chat = re.sub("[^a-zA-Z:;_\s)(]", "", chat)
        chat = word_tokenize(chat)
        chat = [lemmatizer.lemmatize(i) for i in chat if i not in stop_words]
        chat = " ".join(chat) if len(chat) > 0 else np.nan

        df.loc[i, "chat"] = chat

    df = df.dropna()
    df = df.replace({"sentiment": labels_to_idx})
    return df


class Sentiment_Dataloader(pl.LightningDataModule):
    """Dataloader used to load the data for the training the liveness detection and the group loss models which use
    CelebA or LFW, and CFW respectively."""

    def __init__(self, data_src_file, batch_size, max_len, tokenizer,
                 num_workers=4):
        """
        :param name: Choice from DATASETS
        :param batch_size: batch size of the dataloader
        :param num_workers: number of workers to be used to load the data.
        :param input_shape: (Tuple) (channels, width, height).
        :param image_aug_p: if >0 image_aug_p*len(dataset) images will be augmented and added.
        """
        super().__init__()

        self.data_src_file = data_src_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        torch.manual_seed(0)

    def setup(self, stage=None):
        df = load_data(self.data_src_file, column_names=["user_id", "source", "sentiment", "chat"])
        df = preprocess(df)

        df_train = df[df.source.isin(df.source.unique()[:24])][:500]
        df_val = df[df.source.isin(df.source.unique()[24:28])][:100]
        df_test = df[df.source.isin(df.source.unique()[28:])][:100]

        print("number of train, val, test samples", len(df_train), len(df_val), len(df_test))

        self.train_dataset = ChatDataset(df_train["chat"].tolist(), df_train["sentiment"].tolist(), self.tokenizer,
                                         self.max_len)
        self.val_dataset = ChatDataset(df_val["chat"].tolist(), df_val["sentiment"].tolist(), self.tokenizer,
                                       self.max_len)
        self.test_dataset = ChatDataset(df_test["chat"].tolist(), df_test["sentiment"].tolist(), self.tokenizer,
                                        self.max_len)

    # return the dataloader for each split
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True,
                                           sampler=None,
                                           collate_fn=None
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=None
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False,
                                           collate_fn=None
                                           )

    def nb_classes(self):
        return self.train_dataset.nb_classes()


class ChatDataset(Dataset):
    def __init__(self, chats, targets, tokenizer, max_len):
        self.chats = chats
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.chats)

    def __getitem__(self, item):
        chat = str(self.chats[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            chat,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'chat_text': chat,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
