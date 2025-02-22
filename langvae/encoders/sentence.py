import logging
import torch
import torch.nn.functional as F
from typing import Dict
from torch import nn, Tensor
from xxhash import xxh128_digest
from transformers import AutoTokenizer, PreTrainedTokenizer
from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput
from .automodel_presets import AutoModelPreset, PoolingMethod, AUTOMODEL_MAP

logger = logging.getLogger(__name__)


def mean_pooling(model_output: Tensor, attention_mask: Tensor):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def last_token_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
    token_embeddings = model_output.last_hidden_state
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        result = token_embeddings[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = token_embeddings.shape[0]
        result = token_embeddings[torch.arange(batch_size, device=token_embeddings.device), sequence_lengths]

    return result


def cls_token_pooling(model_output: Tensor, attention_mask: Tensor) -> Tensor:
    return model_output.last_hidden_state[:, 0]


class SentenceEncoder(BaseEncoder):
    """
    Encoder class for producing sentence embeddings.

    This encoder uses a pre-trained transformer model to encode input sentences and then projects the encoded
    representations into a latent space defined by the specified latent size. The encoder is designed to work
    with one-hot encoded token tensors and utilizes a linear transformation to produce mean and log-variance
    for variational inference.
    The input tokenization is given by the tokenizer of the decoder model selected for the VAE model where this
    encoder will be connected to. So, for example in a T5-Llama setup the inputs are tokenized using the Llama
    tokenizer. SentenceEncoder takes care of converting the token representations.
    Inputs are given as tensors (B x S x V), where :math:`B` is the batch size, :math:`S` is the maximum sentence
    length and :math:`V` is the decoder vocabulary size.


    Attributes:
        encoder (AutoModel): A HuggingFace pre-trained transformer model.
        linear (nn.Linear): Linear layer to project encoder outputs to latent space.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the encoder model.
        decoder_tokenizer (PreTrainedTokenizer): Tokenizer used for decoding input tokens.
        caching (bool): Whether the pre-trained encoder embeddings will be cached (in memory) or not.
        automodel_preset (dict): Predefined settings for encoder models: Automodel loader class, pooling method, normalization.
        device (str): Device on which the model and data are allocated (e.g., 'cpu', 'cuda').
        debug (bool): Flag to enable/disable debugging output.
    """

    def __init__(self, model_path: str,
                 latent_size: int,
                 decoder_tokenizer: PreTrainedTokenizer,
                 caching: bool = False,
                 automodel_preset: dict = None,
                 device: str = "cpu",
                 args=None):  # Args is a ModelConfig instance
        """
        Initializes the SentenceEncoder with a specified model, latent size, tokenizers, and device.

        Args:
            model_path (str): Path/locator to the pre-trained model.
            latent_size (int): Size of the latent space.
            decoder_tokenizer (PreTrainedTokenizer): Tokenizer for decoding input tensors.
            caching (bool): Activate caching of the pooled pretrained encodings to accelerate processing.
            automodel_preset (dict): Predefined settings for encoder models: Automodel loader class, pooling method, normalization.
            device (str): Device to allocate model and data (e.g., 'cpu', 'cuda').
            args (ModelConfig, optional): Additional configuration arguments.
        """
        BaseEncoder.__init__(self)
        self.model_path = model_path
        self.latent_size = latent_size
        self.device = device
        self._encoder = []
        self._tokenizer = []
        self._decoder_tokenizer = [decoder_tokenizer]
        self.linear = nn.LazyLinear(2 * latent_size, bias=False, device=device)
        self.caching = caching
        self.cache = dict()
        if (automodel_preset):
            self.automodel_preset = AutoModelPreset(**automodel_preset)
        else:
            if (model_path in AUTOMODEL_MAP):
                self.automodel_preset = AutoModelPreset(**AUTOMODEL_MAP[model_path])
            else:
                self.automodel_preset = AutoModelPreset()

        self.pooling = {
            PoolingMethod.MEAN: mean_pooling,
            PoolingMethod.LAST: last_token_pooling,
            PoolingMethod.CLS: cls_token_pooling
        }[self.automodel_preset.pooling_method]

        self.init_pretrained_model()

        # Logging reencoded inputs
        self._dbg_counter = 0
        self.debug = False
        self.output_log_filepath = f"langvae_encoder_{model_path.replace('/', '--')}[{latent_size}].txt"

    @property
    def encoder(self) -> nn.Module:
        return self._encoder[0]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer[0]

    @property
    def decoder_tokenizer(self) -> PreTrainedTokenizer:
        return self._decoder_tokenizer[0]

    @property
    def annotated(self) -> bool:
        return self._encoder[0].config.hidden_size != self.linear.weight.shape[1]

    def to(self, device, include_pretrained: bool = True):
        super().to(device)
        self.device = device
        if (self._encoder and include_pretrained):
            self._encoder[0].to(device)

    def init_pretrained_model(self):
        automodel_cls = self.automodel_preset.cls_type
        self._encoder = [automodel_cls.from_pretrained(self.model_path).to(self.device)]
        self._tokenizer = [AutoTokenizer.from_pretrained(self.model_path)]
        self._encoder[0].eval()
        self._encoder[0].requires_grad_(False)

    def recode(self, tok_ids: list[Tensor] | Tensor) -> Tensor:
        pooled = None
        if (self.caching):
            try:
                pooled_cache = list()
                for i in range(len(tok_ids)):
                    cache_key = xxh128_digest(tok_ids[i].cpu().numpy().data.tobytes(), seed=0)
                    pooled_cache.append(self.cache[cache_key])

                pooled = torch.stack(pooled_cache).to(self.device)
            except KeyError:
                pass

        if (pooled is None):
            input = self.decoder_tokenizer.batch_decode(tok_ids, clean_up_tokenization_spaces=False,
                                                        skip_special_tokens=True)
            enc_toks = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
            enc_attn_mask = enc_toks["attention_mask"].to(self.device)
            enc_toks_ids = enc_toks["input_ids"].to(self.device)
            encoded = self.encoder(input_ids=enc_toks_ids, attention_mask=enc_attn_mask)
            pooled = self.pooling(encoded, enc_attn_mask).detach()
            if (self.automodel_preset.normalize):
                pooled = F.normalize(pooled, p=2, dim=1)

            if (self.caching):
                for i in range(len(tok_ids)):
                    cache_key = xxh128_digest(tok_ids[i].cpu().numpy().data.tobytes(), seed=0)
                    if (cache_key not in self.cache):
                        self.cache[cache_key] = pooled[i]

        return pooled

    def forward(self, x: Tensor, c: Dict[str, Tensor] = None) -> ModelOutput:
        """
        Processes the input tensor through the encoder and linear transformation to produce latent variables.

        Args:
            x (Tensor): Input tensor containing token IDs. Sparse tensors will be interpreted as one-hot.
            c (Tensor): Conditional variable.

        Returns:
            ModelOutput: Object containing the latent embedding and log covariance.
        """

        # Fix for pythae device allocation bug
        self._encoder[0] = self._encoder[0].to(self.device)
        self.linear = self.linear.to(self.device)

        tok_ids = x
        if (x.layout == torch.sparse_coo):
            tok_ids = [x[i].coalesce().indices()[1] for i in range(x.shape[0])]

        uninit = (isinstance(self.linear, nn.LazyLinear) and self.linear.has_uninitialized_params())
        annotatable = uninit or self.annotated
        if (c and not annotatable):
            logger.warning("Passing annotations to a non-annotated model! Annotations will be ignored.")

        pooled = self.recode(tok_ids)
        pooled_cvars = list()
        if (c and annotatable):
            # pooled_cvars = [self.recode(c[annot].to_dense().argmax(dim=-1)) for annot in c]
            pos = torch.linspace(0.1, 1, x.shape[1]).to(self.device)
            pooled_cvars = [
                ((pos * c[annot].to(self.device).mT).mT.sum(dim=-2) / x.shape[1]).to_dense()
                for annot in c
            ]

        enc = torch.cat([pooled] + pooled_cvars, dim=-1) if (c and annotatable) else pooled
        mean, logvar = self.linear(enc).chunk(2, -1)
        output = ModelOutput(
            embedding=mean,
            cvars_embedding=pooled_cvars,
            log_covariance=logvar
        )

        # Debug print (inputs)
        if (self.debug and not self.caching):
            if (self._dbg_counter % 100 == 0):
                input = self.decoder_tokenizer.batch_decode(tok_ids, clean_up_tokenization_spaces=False,
                                                            skip_special_tokens=True)
                enc_toks = self.tokenizer(input, padding=True, truncation=True, return_tensors='pt')
                with open(self.output_log_filepath, "w", encoding="utf-8") as enc_log_file:
                    # print("\n".join(input[:2]))
                    print("\n".join(self.tokenizer.batch_decode(enc_toks["input_ids"])), file=enc_log_file)
                    print("\n", "-" * 40, "\n", file=enc_log_file)
            self._dbg_counter += 1

        return output
