import json
import torch
import torch.nn.functional as F
import jsonlines
from typing import Tuple, List, Union, Iterable
from string import ascii_letters
from itertools import combinations
from xxhash import xxh128_digest
from tqdm import tqdm
from torch.utils.data import Dataset
from pythae.data.datasets import DatasetOutput
from transformers import PreTrainedTokenizer
from saf import Sentence


def get_hash(value: str) -> bytes:
    return xxh128_digest(value.encode("utf8"), seed=0)


class TokenizedDataSet(Dataset):
    """
    A dataset class that handles the tokenization of text data.

    This class is designed to convert text data into a tokenized format suitable for model training or evaluation.
    It supports tokenization of plain string data or structured SAF Sentence objects. The tokenized output is
    converted into a one-hot encoded format for use in neural network models.

    Attributes:
        source (Union[Iterable[Sentence], List[str]]): The source data containing SAF Sentences or strings.
        tokenizer (PreTrainedTokenizer): The tokenizer used for converting text to tokens.
        max_len (int): The maximum length of the tokenized sequences.
        device (str): The computing device (e.g., 'cpu', 'cuda') where the tensors will be stored when retrieved.
    """
    def __init__(self, source: Union[Iterable[Sentence], List[str]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 caching: bool = False,
                 cache_persistence: str = None,
                 sparse: bool = False,
                 device: str = "cpu"):
        """
        Initializes the TokenizedDataSet with the given source data, tokenizer, maximum sequence length, and device.

        Args:
            source (Union[Iterable[Sentence], List[str]]): The source data containing sentences or strings.
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_len (int): The maximum length of the tokenized output.
            caching (bool): Activate caching of the tokenized inputs to accelerate reads.
            cache_persistence (str): File path for persisting cached inputs, if caching is activated.
            sparse (bool): If true, output will be sparse COO tensors, instead of dense ones.
            device (str): The device to which tensors will be sent. Defaults to "cpu".
        """
        self.source = source
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.caching = caching
        self.device = device
        self.cache = dict()
        self.cache_persistence = cache_persistence
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.sparse = sparse

        if (caching and cache_persistence):
            try:
                with jsonlines.open(cache_persistence) as cache_reader:
                    for entry in tqdm(cache_reader, desc=f"Loading dataset cache at {cache_persistence}"):
                        key = bytes.fromhex(entry["key"])
                        value = json.loads(entry["value"])
                        indices = list(range(max_len))
                        self.cache[key] = torch.sparse_coo_tensor([indices, value["ids"]],
                                                                  [1] * len(indices),
                                                                  (max_len, self.vocab_size),
                                                                  dtype=torch.int8)
            except IOError:
                pass

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.source)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index and returns the tokenized and one-hot encoded data.

        Args:
            idx (int or slice): The index of the item to retrieve.

        Returns:
            DatasetOutput: The tokenized and one-hot encoded data as a DatasetOutput object.
        """
        sentences = self.source[idx]
        if isinstance(idx, slice):
            if isinstance(self.source[0], Sentence):
                sentences = [sent.surface for sent in sentences]
        else:
            if isinstance(sentences, Sentence):
                sentences = [sentences.surface]
            else:
                sentences = [sentences]

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        one_hot = None
        keys = None
        if (self.caching):
            keys = [get_hash(sent) for sent in sentences]
            try:
                cached = [self.cache[key] for key in keys]
                one_hot = torch.stack(cached)
            except KeyError:
                pass

        if (one_hot is None):
            tokenized = self.tokenizer(sentences, padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt')
            one_hot = torch.stack([
                torch.sparse_coo_tensor([list(range(self.max_len)), tokenized["input_ids"][i]],
                                        [1] * self.max_len, (self.max_len, self.vocab_size), dtype=torch.int8)
                for i in range(len(tokenized["input_ids"]))
            ])

            if (self.caching):
                for i in range(len(sentences)):
                    if (keys[i] not in self.cache):
                        self.cache[keys[i]] = one_hot[i].coalesce()
                        if (self.cache_persistence):
                            value = {"ids": self.cache[keys[i]].indices()[1].tolist()}
                            with jsonlines.open(self.cache_persistence, mode='a') as cache_writer:
                                cache_writer.write({"key": keys[i].hex(), "value": json.dumps(value)})

        one_hot = one_hot.to(self.device)

        return DatasetOutput(data=one_hot.to_dense() if not self.sparse else one_hot)


class TokenizedAnnotatedDataSet(TokenizedDataSet):
    """
    A dataset class that handles tokenization of text data with annotations.

    This class extends TokenizedDataSet to include handling of annotations alongside the tokenization of text.
    It supports both simple lists of SAF Sentence objects and tuples of sentences with corresponding annotations.

    Attributes:
        source (Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]]): The source data containing annotated sentences.
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        max_len (int): The maximum length of the tokenized output.
        annotations (List[str]): List of annotation types to be processed.
        device (str): The device to which tensors will be sent. Defaults to "cpu".
        annot_map (dict): A mapping from annotation labels to token IDs.
    """
    def __init__(self, source: Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 annotations: List[str],
                 caching: bool = False,
                 cache_persistence: str = None,
                 device: str = "cpu"):
        """
        Initializes the TokenizedAnnotatedDataSet.

        Args:
            source (Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]]): The source data containing annotated sentences.
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_len (int): The maximum length of the tokenized output.
            annotations (List[str]): List of annotation types to be processed.
            caching (bool): Activate caching of the tokenized inputs to accelerate reads.
            cache_persistence (str): File path for persisting cached inputs, if caching is activated.
            device (str): The device to which tensors will be sent. Defaults to "cpu".
        """
        super().__init__(source, tokenizer, max_len, device)
        self.annotations = annotations
        self.annot_map = dict()
        self.cache = dict()
        self.cache_persistence = cache_persistence

        all_labels = {annot: set() for annot in annotations}
        if isinstance(self.source, tuple):
            for item in self.source[1]:
                all_labels[self.annotations[0]].update(item)
        else:
            for sent in self.source:
                for annot in self.annotations:
                    if annot in sent.annotations:
                        all_labels[annot].update(sent.annotations.get(annot, " ") if isinstance(sent.annotations[annot], Iterable) else [sent.annotations[annot]])
                    else:
                        all_labels[annot].update([tok.annotations.get(annot, " ") for tok in sent.tokens])

        char_pairs = sorted([" "] + ["".join(pair) for pair in list(combinations(list(ascii_letters), 2))])
        char_pair_ids = [tok_ids[0] for tok_ids in self.tokenizer(char_pairs, add_special_tokens=False)["input_ids"] if len(tok_ids) < 2]
        for annot in annotations:
            self.annot_map[annot] = dict(zip(sorted(all_labels[annot]), char_pair_ids[:len(all_labels[annot])]))

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.source)

    def __getitem__(self, idx):
        """
        Retrieves an item by its index, including annotations, and returns the tokenized, one-hot encoded data and annotations as a DatasetOutput object.

        Args:
            idx (int or slice): The index of the item to retrieve.

        Returns:
            DatasetOutput: The tokenized, one-hot encoded data and annotations as a DatasetOutput object.
        """
        if isinstance(self.source, tuple):
            sentences = self.source[0][idx] if isinstance(idx, slice) else [self.source[0][idx]]
            labels = {self.annotations[0]: self.source[1][idx]}
        else:
            items = self.source[idx] if isinstance(idx, slice) else [self.source[idx]]
            sentences = [[tok.surface for tok in sent.tokens] for sent in items]
            labels = {annot: list() for annot in self.annotations}
            for sent in items:
                for annot in self.annotations:
                    if annot in sent.annotations:
                        labels[annot].append(sent.annotations[annot])
                    else:
                        labels[annot].append([tok.annotations.get(annot, " ") for tok in sent.tokens])

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prefixing with space for word indexing on new tokenizers (Llama3, Gemma, etc.)
        for i in range(len(sentences)):
            sentences[i] = [" " + tok for tok in sentences[i]]

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
