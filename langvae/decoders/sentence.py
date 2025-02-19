import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, DynamicCache
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput

FLASH_ATTN_SUPPORTED = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-3B"
]


class SentenceDecoder(BaseDecoder):
    """
    Decoder class for generating sentences from latent representations.

    This decoder uses a pre-trained causal language model to generate text from latent representations.
    It outputs token probability distribution tensors (B x S x V), where :math:`B` is the batch size, :math:`S`
    is the maximum sentence length and :math:`V` is the decoder vocabulary size.

    Attributes:
        model_path (str): Path/locator to the pre-trained language model.
        latent_size (int): Size of the latent space.
        max_len (int): Maximum length (in tokens) of the generated sentences.
        device (torch.device): Device on which the model and data are allocated (e.g., 'cpu', 'cuda').
        device_map (str): Device map configuration for model parallelism.
        args (ModelConfig, optional): Additional configuration arguments.
    """

    def __init__(self, model_path: str,
                 latent_size: int,
                 max_len: int,
                 conditional: bool = False,
                 device: torch.device = "cpu",
                 device_map: str = None,
                 args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)
        self.model_path = model_path
        self.latent_size = latent_size

        self.max_len = max_len
        self.conditional = conditional
        self.device_map = device_map if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else None
        self.device = device
        self.dec_hidden_layer_dev_map = None
        self._decoder = []
        self._tokenizer = []

        self.init_pretrained_model()

        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.pad_token_id] * 2, dtype=torch.int64, device=self.device), dim=-1)
        pkv = self.decoder(dec_ids, use_cache=True).past_key_values
        self.pkv_dims = pkv[0][0].shape[1:]
        self.pkv_dtype = pkv[0][0].dtype

        self.context_hidden = nn.ModuleList([
            nn.LazyLinear(
                self.pkv_dims[0] * self.pkv_dims[1] * self.pkv_dims[2] * 2 * (self.max_len + 1), # self.pkv_dims[0] * self.pkv_dims[1] * self.pkv_dims[2] == self.decoder.config.hidden_size
                dtype=self.pkv_dtype,
                device=f"cuda:{self.dec_hidden_layer_dev_map[i]}" if self.dec_hidden_layer_dev_map else self.device
            )
            for i in range(self.decoder.config.num_hidden_layers)
        ])

        self.dropout = nn.Dropout(p=0.1)

        # Logging outputs
        self._dbg_counter = 0
        self.debug = False
        self.output_log_filepath = f"langvae_decoder_{model_path.replace('/', '--')}[{latent_size}_{max_len}].txt"

    @property
    def decoder(self) -> nn.Module:
        return self._decoder[0]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer[0]

    def to(self, device, include_pretrained: bool = True):
        super().to(device)
        self.device = device
        self.dec_hidden_layer_dev_map = None
        if (self._decoder and include_pretrained):
            self._decoder[0].to(device)


    def init_pretrained_model(self):
        ex_params = dict()
        if (self.model_path in FLASH_ATTN_SUPPORTED):
            ex_params = {"attn_implementation": "flash_attention_2", "offload_buffers": True}
        if (str(self.device).startswith("cuda") and self.device_map):
            self._decoder = [AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto",
                                                                  device_map=self.device_map, **ex_params)]
            dec_hidden_layer_prefix = [".".join(layer.split(".")[:-1]) for layer in self._decoder[0].hf_device_map
                                       if (layer.split(".")[-1] == str(self.decoder.config.num_hidden_layers - 1))
                                       ][0]
            self.dec_hidden_layer_dev_map = {i: self._decoder[0].hf_device_map[f"{dec_hidden_layer_prefix}.{i}"]
                                             for i in range(self.decoder.config.num_hidden_layers)}
        else:
            self._decoder = [AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto")]
            self._decoder = [self.decoder.to(self.device)]

        self._decoder[0].eval()
        self._decoder[0].requires_grad_(False)

        self._tokenizer = [AutoTokenizer.from_pretrained(self.model_path, padding_side="left", add_prefix_space=True)]
        self._tokenizer[0].pad_token = self.tokenizer.eos_token
        self._tokenizer[0].pad_token_id = self.tokenizer.eos_token_id
        self._tokenizer[0].bos_token_id = (self._tokenizer[0].bos_token_id or self._decoder[0].config.bos_token_id)
        self.device = self.decoder.device

    def forward(self, z: Tensor, max_len: int = 0) -> ModelOutput:
        """
        Processes the input latent tensor through the decoder to generate sentences.

        Args:
            z (Tensor): Input tensor containing latent representations.
            max_len (int): Maximum length (tokens) of output sentences.

        Returns:
            ModelOutput: The generated sentences as a ModelOutput object: token probability distribution
            tensors (B x S x V), where :math:`B` is the batch size, :math:`S is the maximum sentence length and
            :math:`V` is the decoder vocabulary size.
        """
        # Fix for pythae device allocation bug
        if (not self.device_map):
            self._decoder[0] = self._decoder[0].to(self.device)
        else:
            self.device = self._decoder[0].device
            # self.context_embedder = self.context_embedder.to(self.device)

        if (not max_len):
            max_len = self.max_len

        z = z.to(self.pkv_dtype).to(self.device)
        dev_map = self.dec_hidden_layer_dev_map

        z_repl = None
        if (dev_map):
            for layer_idx in range(len(self.context_hidden)):
                self.context_hidden[layer_idx].to(f"cuda:{dev_map[layer_idx]}")
            z_repl = {dev: z.to(f"cuda:{dev}") for dev in set(dev_map.values())}

        generated = torch.zeros(z.shape[0], max_len + 1, self.decoder.config.vocab_size, device=self.device, dtype=self.pkv_dtype)
        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.bos_token_id] * z.shape[0], dtype=torch.int64, device=self.device), dim=-1)
        # decoded = self.decoder(input_ids=dec_ids)
        # generated[:, 0, :] = F.softmax(decoded.logits[:, -1, :], dim=-1)
        generated[:, 0,:] += F.one_hot(dec_ids, num_classes=generated.shape[-1]).squeeze()
        ctx_mem = [None] * self.decoder.config.num_hidden_layers
        past_dec = [None] * self.decoder.config.num_hidden_layers

        for layer_idx in range(self.decoder.config.num_hidden_layers):
            hidden_state = z_repl[dev_map[layer_idx]] if dev_map else z

            past = tuple([
                h.view(-1,
                       self.pkv_dims[0],
                       self.pkv_dims[1] * (self.max_len + 1),
                       self.pkv_dims[2]).to(f"cuda:{dev_map[layer_idx]}" if dev_map else self.device)
                for h in self.dropout(self.context_hidden[layer_idx](hidden_state)).chunk(2, dim=-1)
            ])

            ctx_mem[layer_idx] = (
                past[0],
                past[1]
            )
            past_dec[layer_idx] = (
                past[0][:, :, :1, :],
                past[1][:, :, :1, :]
            )

        for i in range(max_len):
            # ctx_embed = context_embeds[:, max(0, i-self.max_look_behind + 1):i+1, :]
            # past_dec = self.compute_kv_residuals(decoded.past_key_values, z, z_repl, dev_map)
            gen_ids = generated[:, max(0, i):i+1, :].argmax(dim=-1)
            # embeds = self.decoder.get_input_embeddings()(gen_ids)  # + ctx_embed
            past_dec = DynamicCache.from_legacy_cache(past_dec)
            decoded = self.decoder(input_ids=gen_ids, use_cache=True, past_key_values=past_dec)

            past_dec = [
                (
                    torch.cat([
                        decoded.past_key_values[layer_idx][0][:, :, :-2, :],
                        ctx_mem[layer_idx][0][:, :, i+1:i+2, :],
                        decoded.past_key_values[layer_idx][0][:, :, -1:, :]
                    ],
                    dim=-2),
                    torch.cat([
                        decoded.past_key_values[layer_idx][1][:, :, :-2, :],
                        ctx_mem[layer_idx][1][:, :, i+1:i+2, :],
                        decoded.past_key_values[layer_idx][1][:, :, -1:, :]
                    ],
                    dim=-2),
                )
                for layer_idx in range(self.decoder.config.num_hidden_layers)
            ]

            generated[:, i+1, :] = F.softmax(decoded.logits[:, -1, :], dim=-1)

        # Debug print (outputs)
        if (self.debug):
            if (self._dbg_counter % 100 == 0):
                with open(self.output_log_filepath, "w", encoding="utf-8") as output_log_file:
                    print(
                        "\n".join([s.replace(self.tokenizer.pad_token, "|#|")
                                   for s in self.tokenizer.batch_decode(torch.argmax(generated, dim=-1))]),
                        file=output_log_file
                    )
                    print("\n", "-" * 40, "\n", file=output_log_file)

            self._dbg_counter += 1

        output = ModelOutput(
            reconstruction=generated[:, 1:max_len + 1,:]
        )
        return output
