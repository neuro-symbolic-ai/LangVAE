import torch
from enum import Flag, auto
from torch import nn, Tensor
from transformers import PreTrainedTokenizer
from pythae.models.base.base_utils import ModelOutput
from .sentence import SentenceEncoder, mean_pooling


class AnnotationType(Flag):
    TOKEN = auto()
    SENTENCE = auto()


class AnnotatedSentenceEncoder(SentenceEncoder):
    def __init__(self, model_path: str,
                 latent_size: int,
                 decoder_tokenizer: PreTrainedTokenizer,
                 num_annotations: int,
                 device: str = "cpu",
                 args=None):  # Args is a ModelConfig instance
        super().__init__(model_path, latent_size, decoder_tokenizer, device, args)
        self.num_annotations = num_annotations
        bottleneck_size = self.encoder.config.hidden_size * (num_annotations + 1)
        self.linear = nn.Linear(bottleneck_size, 2 * latent_size, bias=False, device=device)

    def forward(self, x: Tensor) -> ModelOutput:
        # Fix for pythae device allocation bug
        self.encoder = self.encoder.to(self.device)
        self.linear = self.linear.to(self.device)

        pooled_annots = list()
        x_split = torch.squeeze(x).to(self.device).chunk(self.num_annotations + 1, dim=-1)
        x_tokens = x_split[0]
        tok_ids = torch.argmax(x_tokens, dim=-1)
        input = self.decoder_tokenizer.batch_decode(tok_ids, clean_up_tokenization_spaces=True,
                                                    skip_special_tokens=True)
        enc_toks = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        enc_attn_mask = enc_toks["attention_mask"].to(self.device)
        encoded = self.encoder(enc_toks["input_ids"].to(self.device), attention_mask=enc_attn_mask)
        pooled = mean_pooling(encoded, enc_attn_mask)

        for lbl_split in x_split[1:]:
            lbl_ids = torch.argmax(lbl_split, dim=-1)
            input = self.decoder_tokenizer.batch_decode(lbl_ids, clean_up_tokenization_spaces=True,
                                                        skip_special_tokens=True)
            enc_lbls = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
            enc_lbl_attn_mask = enc_lbls["attention_mask"].to(self.device)
            encoded_lbls = self.encoder(enc_lbls["input_ids"].to(self.device), attention_mask=enc_lbl_attn_mask)
            pooled_lbls = mean_pooling(encoded_lbls, enc_lbl_attn_mask)
            pooled_annots.append(pooled_lbls)

        mean, logvar = self.linear(torch.cat([pooled] + pooled_annots, dim=-1)).chunk(2, -1)

        output = ModelOutput(
            embedding=mean,
            log_covariance=logvar,
            embedding_lbls=pooled_annots
        )

        # Debug print (inputs)
        if (self.debug):
            if (self.dbg_counter % 100 == 0):
                print()
                # print("\n".join(input[:2]))
                print("\n".join(self.tokenizer.batch_decode(enc_toks["input_ids"])))
            self.dbg_counter += 1

        return output
