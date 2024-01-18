import torch
import torch.nn.functional as F
from typing import List, Union, Iterable
from torch.utils.data import Dataset
from pythae.data.datasets import DatasetOutput
from transformers import PreTrainedTokenizer
from saf import Sentence


class TokenizedDataSet(Dataset):
    def __init__(self,  source: Union[Iterable[Sentence], List[str]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 device: str = "cpu"):
        self.source = source
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sentences = self.source[idx]
        if (isinstance(idx, slice)):
            if (isinstance(self.source[0], Sentence)):
                sentences = [sent.surface for sent in sentences]
        else:
            if (isinstance(sentences, Sentence)):
                sentences = [sentences.surface]
            else:
                sentences = [sentences]

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        tokenized = self.tokenizer(sentences, padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')
        one_hot = F.one_hot(tokenized["input_ids"], num_classes=len(self.tokenizer.get_vocab())).to(torch.int8)

        return DatasetOutput(data=one_hot.to(self.device))
