import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


class SentenceEncoder(BaseEncoder):
    def __init__(self, model_path: str,
                 latent_size: int,
                 decoder_tokenizer: PreTrainedTokenizer,
                 device: str = "cpu",
                 args=None):  # Args is a ModelConfig instance
        BaseEncoder.__init__(self)
        self.encoder = AutoModel.from_pretrained(model_path).to(device)
        self.linear = nn.Linear(self.encoder.config.hidden_size, 2 * latent_size, bias=False, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder.eval()
        self.device = device
        self.dbg_counter = 0

    def forward(self, x: Tensor) -> ModelOutput:
        x = torch.squeeze(x).to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.linear = self.linear.to(self.device)
        tok_ids = torch.argmax(x, dim=-1)
        input = self.decoder_tokenizer.batch_decode(tok_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        enc_toks = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
        enc_attn_mask = enc_toks["attention_mask"].to(self.device)

        encoded = self.encoder(enc_toks["input_ids"].to(self.device), attention_mask=enc_attn_mask)
        pooled = mean_pooling(encoded, enc_attn_mask)
        mean, logvar = self.linear(pooled).chunk(2, -1)
        output = ModelOutput(
            embedding=mean,
            log_covariance=logvar
        )

        # Debug print (inputs)
        if (self.dbg_counter % 100 == 0):
            print()
            # print("\n".join(input[:2]))
            print("\n".join(self.tokenizer.batch_decode(enc_toks["input_ids"])))
        self.dbg_counter += 1

        return output
