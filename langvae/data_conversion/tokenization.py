import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Iterable
from string import ascii_letters
from itertools import combinations
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

class TokenizedAnnotatedDataSet(TokenizedDataSet):
    def __init__(self, source: Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 annotations: List[str],
                 device: str = "cpu"):
        self.source = source
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.annotations = annotations
        self.device = device
        self.annot_map = dict()

        all_labels = {annot: set() for annot in annotations}
        if (isinstance(self.source, tuple)):
            for item in self.source[1]:
                all_labels[self.annotations[0]].update(item)
        else:
            for sent in self.source:
                for annot in self.annotations:
                    if (annot in sent.annotations):
                        all_labels[annot].update(sent.annotations.get(annot, " ") if isinstance(sent.annotations[annot], Iterable) else [sent.annotations[annot]])
                    else:
                        all_labels[annot].update([tok.annotations.get(annot, " ") for tok in sent.tokens])

        char_pairs = sorted([" "] + ["".join(pair) for pair in list(combinations(list(ascii_letters), 2))])
        char_pair_ids = [tok_ids[0] for tok_ids in self.tokenizer(char_pairs)["input_ids"] if len(tok_ids) < 2]
        for annot in annotations:
            self.annot_map[annot] = dict(zip(sorted(all_labels[annot]), char_pair_ids[:len(all_labels[annot])]))


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if (isinstance(self.source, tuple)):
            sentences = self.source[0][idx] if (isinstance(idx, slice)) else [self.source[0][idx]]
            labels = {self.annotations[0]: self.source[1][idx]}
        else:
            items = self.source[idx] if (isinstance(idx, slice)) else [self.source[idx]]
            sentences = [[tok.surface for tok in sent.tokens] for sent in items]
            labels = {annot: list() for annot in self.annotations}
            for sent in items:
                for annot in self.annotations:
                    if (annot in sent.annotations):
                        labels[annot].append(sent.annotations[annot])
                    else:
                        labels[annot].append([tok.annotations.get(annot, " ") for tok in sent.tokens])

        if (not self.tokenizer.pad_token):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        tokenized = self.tokenizer(sentences, padding="max_length", truncation=True, max_length=self.max_len,
                                   is_split_into_words=True, return_tensors='pt')
        one_hot = F.one_hot(tokenized["input_ids"], num_classes=len(self.tokenizer.get_vocab())).to(torch.int8)
        words_idx = [[widx for widx in tokenized.words(i) if widx is not None]
                     for i in range(tokenized["input_ids"].shape[0])]

        tokenized_annot_oh = dict()
        for annot in self.annotations:
            tokenized_annot_oh[annot] = [F.one_hot(torch.tensor([self.annot_map[annot][lbl] for lbl in lbls]),
                                                   num_classes=len(self.tokenizer.get_vocab())).to(torch.int8)
                                         for lbls in labels[annot]]

        batch = torch.zeros((one_hot.shape[0], one_hot.shape[1], one_hot.shape[2] * (len(self.annotations) + 1)))
        for i in range(one_hot.shape[0]):
            for j in range(one_hot.shape[1] - len(words_idx[i])):
                batch[i,j][:one_hot.shape[2]] = one_hot[i,j]
                for k in range(len(self.annotations)):
                    batch[i, j][one_hot.shape[2] * (k + 1): one_hot.shape[2] * (k + 2)] = one_hot[i,j]
            for j in range(one_hot.shape[1] - len(words_idx[i]), one_hot.shape[1]):
                batch[i, j][:one_hot.shape[2]] = one_hot[i, j]
                for k in range(len(self.annotations)):
                    annot = self.annotations[k]
                    p = j - (one_hot.shape[1] - len(words_idx[i]))
                    try:
                        batch[i, j][one_hot.shape[2] * (k + 1): one_hot.shape[2] * (k + 2)] = tokenized_annot_oh[annot][i][words_idx[i][p]]
                    except:
                        print("ERROR!")

        return DatasetOutput(data=batch.to(self.device))
