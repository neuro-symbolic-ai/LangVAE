import json
import torch
from torch import Tensor
import torch.nn.functional as F
import jsonlines
from typing import Tuple, List, Dict, Union, Iterable
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

def collate_sparse_fn(batch, *, collate_fn_map: dict = None):
    elem: Tensor = batch[0]

    if (isinstance(elem, DatasetOutput)):
        result = dict()
        for key in elem:
            if (isinstance(elem[key], Tensor)):  # Pads tensors to the batch maximum length
                max_len = max([b[key][0].shape[0] for b in batch])
                if (elem[key].layout == torch.sparse_coo):
                    result[key] = torch.stack([
                        torch.cat([
                            b[key][0],
                            torch.zeros((max_len - b[key][0].shape[0],) + tuple(b[key][0].shape[1:]),
                                        dtype=b[key][0].dtype,
                                        layout=b[key][0].layout)
                        ])
                        for b in batch
                    ])
                else:
                    result[key] = torch.stack([
                        torch.cat([
                            b[key][0],
                            torch.zeros((max_len - b[key][0].shape[0],) + tuple(b[key][0].shape[1:]),
                                        dtype=b[key][0].dtype)
                        ])
                        for b in batch
                    ])
            else:
                result[key] = [b[key][0] for b in batch]

        result = DatasetOutput(result)
    else:
        result = batch

    return result


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
        caching (bool): Activate caching of the tokenized inputs to accelerate reads.
        cache_persistence (str): File path for persisting cached inputs, if caching is activated.
        tokenizer_options (dict): Options for the tokenizer.
        vocab_size (int): Size of the tokenizer vocabulary.
    """
    def __init__(self, source: Union[Iterable[Sentence], List[str]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 caching: bool = False,
                 cache_persistence: str = None,
                 return_tensors: bool = True,
                 one_hot: bool = True,
                 tokenizer_options: dict = None):
        """
        Initializes the TokenizedDataSet with the given source data, tokenizer and maximum sequence length.

        Args:
            source (Union[Iterable[Sentence], List[str]]): The source data containing sentences or strings.
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_len (int): The maximum length of the tokenized output.
            caching (bool): Activate caching of the tokenized inputs to accelerate reads.
            cache_persistence (str): File path for persisting cached inputs, if caching is activated.
        """
        self.source = source
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.caching = caching
        self.cache = dict()
        self.cache_persistence = cache_persistence.replace("/", "__") if cache_persistence else ""
        self.return_tensors = return_tensors
        self.one_hot = one_hot
        self.tokenizer_options = dict() if not tokenizer_options else tokenizer_options
        self.vocab_size = len(self.tokenizer.get_vocab())

        if ("return_tensors" in self.tokenizer_options):
            del self.tokenizer_options["return_tensors"]

        if (return_tensors and not one_hot):
            self.tokenizer_options["padding"] = True

        if (caching and cache_persistence):
            try:
                with jsonlines.open(cache_persistence) as cache_reader:
                    for entry in tqdm(cache_reader, desc=f"Loading dataset cache at {cache_persistence}"):
                        key = bytes.fromhex(entry["key"])
                        value = json.loads(entry["value"])
                        self.cache[key] = value
            except IOError:
                pass

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of items.
        """
        return len(self.source)

    def __getitem__(self, idx) -> DatasetOutput:
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

        tokenized = None
        keys = None
        if (self.caching):
            keys = [get_hash(sent) for sent in sentences]
            try:
                cached = [self.cache[key] for key in keys]
                tokenized = {"input_ids": [c["input_ids"] for c in cached],
                             "attention_mask": [c["attention_mask"] for c in cached]}
            except KeyError:
                pass

        if (tokenized is None):
            tokenized = self.tokenizer(sentences, truncation=True, max_length=self.max_len, **self.tokenizer_options)

            if (self.caching):
                for i in range(len(sentences)):
                    if (keys[i] not in self.cache):
                        self.cache[keys[i]] = {
                            "input_ids": tokenized["input_ids"][0],
                            "attention_mask": tokenized["attention_mask"][0]
                        }
                        if (self.cache_persistence):
                            value = {"input_ids": self.cache[keys[i]]["input_ids"],
                                     "attention_mask": self.cache[keys[i]]["attention_mask"]}
                            with jsonlines.open(self.cache_persistence, mode='a') as cache_writer:
                                cache_writer.write({"key": keys[i].hex(), "value": json.dumps(value)})

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if (self.one_hot):
            max_len = max([len(ids) for ids in input_ids])
            input_ids = torch.stack([
                torch.cat([
                    torch.sparse_coo_tensor([list(range(len(ids))), ids],
                                            [1] * len(ids),
                                            (len(ids), self.vocab_size),
                                            dtype=torch.int8),
                    torch.zeros(max_len - len(ids), self.vocab_size, dtype=torch.int8, layout=torch.sparse_coo)
                ])
                for ids in input_ids
            ])

            attention_mask = torch.stack([torch.tensor(atts + [0] * (max_len - len(atts))) for atts in attention_mask])

            if (not self.return_tensors):
                input_ids = input_ids.tolist()
        elif (self.return_tensors):
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(tokenized["attention_mask"])

        return DatasetOutput(data=input_ids,
                             input_ids=input_ids,
                             attention_mask=attention_mask)


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
        caching (bool): Activate caching of the tokenized inputs to accelerate reads.
        cache_persistence (str): File path for persisting cached inputs, if caching is activated.
        tokenizer_options (dict): Options for the tokenizer.
    """
    def __init__(self, source: Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]],
                 tokenizer: PreTrainedTokenizer,
                 max_len: int,
                 annotations: Dict[str, List[str]],
                 caching: bool = False,
                 cache_persistence: str = None,
                 return_tensors: bool = True,
                 one_hot: bool = True,
                 tokenizer_options: dict = None):
        """
        Initializes the TokenizedAnnotatedDataSet.

        Args:
            source (Union[Iterable[Sentence], Tuple[List[List[str]], List[List[str]]]]): The source data containing annotated sentences.
            tokenizer (PreTrainedTokenizer): The tokenizer to be used for tokenization.
            max_len (int): The maximum length of the tokenized output.
            annotations (Dict[str, List[str]]): List of annotation types to be processed.
            caching (bool): Activate caching of the tokenized inputs to accelerate reads.
            cache_persistence (str): File path for persisting cached inputs, if caching is activated.
            device (str): The device to which tensors will be sent. Defaults to "cpu".
        """
        super().__init__(source, tokenizer, max_len, caching, cache_persistence,
                         return_tensors, one_hot, tokenizer_options)
        self.annotations = annotations

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
            labels = {list(self.annotations.keys())[0]: self.source[1][idx]}
        else:
            items = self.source[idx] if isinstance(idx, slice) else [self.source[idx]]
            sentences = [[tok.surface for tok in sent.tokens] for sent in items]
            labels = list()
            for sent in items:
                labels.append(dict())
                for annot in self.annotations:
                    if annot in sent.annotations:
                        labels[-1][annot] = sent.annotations[annot]
                    else:
                        labels[-1][annot] = [tok.annotations.get(annot, " ") for tok in sent.tokens]

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prefixing with space for word indexing on new tokenizers (Llama3, Gemma, etc.)
        for i in range(len(sentences)):
            sentences[i] = [" " + tok for tok in sentences[i]]

        tokenized = None
        keys = None
        annotations = None
        if (self.caching):
            keys = [get_hash(" ".join(sent)) for sent in sentences]
            try:
                cached = [self.cache[key] for key in keys]
                tokenized = {"input_ids": [c["input_ids"] for c in cached],
                             "attention_mask": [c["attention_mask"] for c in cached]}
                annotations = [c["annotations"] for c in cached]
            except KeyError:
                pass

        if (tokenized is None):
            tokenized = self.tokenizer(sentences, truncation=True, max_length=self.max_len,
                                       is_split_into_words=True, **self.tokenizer_options)
            words_idx = [[widx for widx in tokenized.words(i) if widx is not None]
                         for i in range(len(tokenized["input_ids"]))]

            annotations = list()
            for lbls in labels:
                annotations.append(dict())
                for annot in lbls:
                    annotations[-1][annot] = [self.annotations[annot].index(lbl) for lbl in lbls[annot]][:self.max_len]

            if (self.caching):
                for i in range(len(sentences)):
                    if (keys[i] not in self.cache):
                        self.cache[keys[i]] = {
                            "input_ids": tokenized["input_ids"][i],
                            "attention_mask": tokenized["attention_mask"][i],
                            "annotations": annotations[i]
                        }
                        if (self.cache_persistence):
                            value = {"input_ids": self.cache[keys[i]]["input_ids"],
                                     "attention_mask": self.cache[keys[i]]["attention_mask"],
                                     "annotations": self.cache[keys[i]]["annotations"]}
                            with jsonlines.open(self.cache_persistence, mode='a') as cache_writer:
                                cache_writer.write({"key": keys[i].hex(), "value": json.dumps(value)})

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if (self.one_hot):
            max_len = max([len(ids) for ids in input_ids])
            input_ids = torch.stack([
                torch.cat([
                    torch.sparse_coo_tensor([list(range(len(ids))), ids],
                                            [1] * len(ids),
                                            (len(ids), self.vocab_size),
                                            dtype=torch.int8),
                    torch.zeros(max_len - len(ids), self.vocab_size, dtype=torch.int8, layout=torch.sparse_coo)
                ])
                for ids in input_ids
            ])

            attention_mask = torch.stack([torch.tensor(atts + [0] * (max_len - len(atts))) for atts in attention_mask])

            annotations_oh = list()
            for i in range(len(annotations)):
                annotations_oh.append(dict())
                for annot in self.annotations:
                    lbl_ids = annotations[i][annot]
                    annotations_oh[i][annot] = torch.cat([
                        torch.sparse_coo_tensor([list(range(len(lbl_ids))), lbl_ids],
                                                [1] * len(lbl_ids),
                                                (len(lbl_ids), len(self.annotations[annot])),
                                                dtype=torch.int8),
                        torch.zeros(max_len - len(lbl_ids), len(self.annotations[annot]), dtype=torch.int8, layout=torch.sparse_coo)
                    ])

            annotations = annotations_oh

            if (not self.return_tensors):
                input_ids = input_ids.tolist()
                annotations_l = list()
                for i in range(len(annotations)):
                    annotations_l.append(dict())
                    for annot in self.annotations:
                        annotations_l[i][annot] = annotations[annot].tolist()

                annotations = annotations_l

        elif (self.return_tensors):
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(tokenized["attention_mask"])
            annotations_t = list()
            for i in range(len(annotations)):
                annotations_t.append(dict())
                for annot in self.annotations:
                    annotations_t[i][annot] = torch.tensor(annotations[annot])

            annotations = annotations_t

        if (self.one_hot or self.return_tensors):
            annotations = {annot: torch.stack([annotations[i][annot] for i in range(len(annotations))])}

        return DatasetOutput(data=input_ids,
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             **annotations)

