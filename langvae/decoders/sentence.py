import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput

FLASH_ATTN_SUPPORTED = [
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.3"
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
        load_in_4bit (bool): Flag indicating whether to load the model in 4-bit quantisation mode for memory efficiency.
        device_map (str): Device map configuration for model parallelism.
        max_look_behind (int): Maximum number of tokens to look behind for context in generation.
        args (ModelConfig, optional): Additional configuration arguments.
    """

    def __init__(self, model_path: str,
                 latent_size: int,
                 max_len: int,
                 device: torch.device = "cpu",
                 load_in_4bit: bool = False,
                 device_map: str = None,
                 max_look_behind: int = 2,
                 args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)
        self.model_path = model_path

        self.max_len = max_len
        self.max_look_behind = max_look_behind
        self.device_map = device_map
        self.device = device
        self.dec_hidden_layer_dev_map = None
        self.load_in_4bit = load_in_4bit
        self._decoder = []
        self._tokenizer = []

        self.init_pretrained_model()

        embedding_dim = self.decoder.get_input_embeddings().embedding_dim

        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.pad_token_id] * 2, dtype=torch.int64, device=self.device), dim=-1)
        pkv = self.decoder(dec_ids).past_key_values
        self.pkv_dims = pkv[0][0].shape[1:]
        self.pkv_dtype = pkv[0][0].dtype

        self.context_embedder = nn.Linear(latent_size, max_len * embedding_dim, dtype=self.pkv_dtype, device=self.device)
        self.context_hidden = nn.ModuleList([
            nn.Linear(
                latent_size,
                self.pkv_dims[0] * self.pkv_dims[1] * self.pkv_dims[2] * 4, # self.pkv_dims[0] * self.pkv_dims[1] * self.pkv_dims[2] == self.decoder.config.hidden_size
                dtype=self.pkv_dtype,
                device=f"cuda:{self.dec_hidden_layer_dev_map[i]}" if self.dec_hidden_layer_dev_map else self.device
            )
            for i in range(self.decoder.config.num_hidden_layers)
        ])
        self.dropout = nn.Dropout(p=0.4)

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

    def init_pretrained_model(self):
        ex_params = dict()
        if (self.model_path in FLASH_ATTN_SUPPORTED):
            ex_params = {"attn_implementation": "flash_attention_2", "offload_buffers": True}
        if (str(self.device).startswith("cuda") and self.device_map):
            self._decoder = [AutoModelForCausalLM.from_pretrained(self.model_path, load_in_4bit=self.load_in_4bit,
                                                                  torch_dtype="auto", device_map=self.device_map,
                                                                  **ex_params)]
            dec_hidden_layer_prefix = [".".join(layer.split(".")[:-1]) for layer in self._decoder[0].hf_device_map
                                       if (layer.split(".")[-1] == str(self.decoder.config.num_hidden_layers - 1))
                                       ][0]
            self.dec_hidden_layer_dev_map = {i: self._decoder[0].hf_device_map[f"{dec_hidden_layer_prefix}.{i}"]
                                             for i in range(self.decoder.config.num_hidden_layers)}
        else:
            self._decoder = [AutoModelForCausalLM.from_pretrained(self.model_path, load_in_4bit=self.load_in_4bit,
                                                                  torch_dtype="auto", **ex_params)]
            if (not self.load_in_4bit):
                self._decoder = [self.decoder.to(self.device)]

        self._decoder[0].eval()
        self._decoder[0].requires_grad_(False)

        self._tokenizer = [AutoTokenizer.from_pretrained(self.model_path, padding_side="left", add_prefix_space=True)]
        self._tokenizer[0].pad_token = self.tokenizer.eos_token
        self._tokenizer[0].pad_token_id = self.tokenizer.eos_token_id
        self.device = self.decoder.device

    def forward(self, z: Tensor) -> ModelOutput:
        """
        Processes the input latent tensor through the decoder to generate sentences.

        Args:
            z (Tensor): Input tensor containing latent representations.

        Returns:
            ModelOutput: The generated sentences as a ModelOutput object: token probability distribution
            tensors (B x S x V), where :math:`B` is the batch size, :math:`S is the maximum sentence length and
            :math:`V` is the decoder vocabulary size.
        """
        # Fix for pythae device allocation bug
        if (not (self.load_in_4bit or self.device_map)):
            self._decoder[0] = self._decoder[0].to(self.device)
        else:
            self.device = self._decoder[0].device
            # if (self.dec_hidden_layer_dev_map):
            #     for i in self.dec_hidden_layer_dev_map:
            #         self.context_hidden[i].to(f"cuda:{self.dec_hidden_layer_dev_map[i]}")
            # else:
            #     self.context_hidden = self.context_hidden.to(self.device)
            self.context_embedder = self.context_embedder.to(self.device)

        z = z.to(self.pkv_dtype).to(self.device)
        dev_map = self.dec_hidden_layer_dev_map

        embedding_dim = self.decoder.get_input_embeddings().embedding_dim
        context_embeds = self.dropout(self.context_embedder(z)).view(-1, self.max_len, embedding_dim)
        past = [
            tuple([h.view(-1,
                          self.pkv_dims[0],
                          self.pkv_dims[1],
                          self.pkv_dims[2]).to(f"cuda:{dev_map[l_idx]}" if dev_map else self.device)
                   for h in self.dropout(self.context_hidden[l_idx](z)).chunk(4, dim=-1)])
            for l_idx in range(len(self.context_hidden))
        ]
        init_past = tuple([(F.tanh(p[0]) + p[1], F.tanh(p[2]) + p[3]) for p in past])

        generated = torch.zeros(z.shape[0], self.max_len + 1, self.decoder.config.vocab_size, device=self.device, dtype=self.pkv_dtype)
        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.pad_token_id] * z.shape[0], dtype=torch.int64, device=self.device), dim=-1)
        decoded = self.decoder(input_ids=dec_ids, past_key_values=init_past)
        generated[:, 0,:] += F.one_hot(dec_ids, num_classes=generated.shape[-1]).squeeze()
        past_dec = [None] * self.decoder.config.num_hidden_layers

        for i in range(self.max_len):
            for layer_idx in range(self.decoder.config.num_hidden_layers):
                past_dec[layer_idx] = (
                    F.tanh(past[layer_idx][0]) * decoded.past_key_values[layer_idx][0][:, :, 1:, :] + past[layer_idx][1],
                    F.tanh(past[layer_idx][2]) * decoded.past_key_values[layer_idx][1][:, :, 1:, :] + past[layer_idx][3]
                )

            # decoded = self.decoder(input_ids=generated[:, max(0, i-self.max_look_behind):i+1, :].argmax(dim=-1),
            #                        past_key_values=tuple(past_dec))
            gen_ids = generated[:, max(0, i-self.max_look_behind):i+1, :].argmax(dim=-1)
            embeds = self.decoder.get_input_embeddings()(gen_ids) + context_embeds[:, max(0, i-self.max_look_behind):i+1, :]
            decoded = self.decoder(inputs_embeds=embeds, past_key_values=tuple(past_dec))
            generated[:, i+1,:] = F.softmax(decoded.logits[:, -1, :], dim=-1)

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
            reconstruction=generated[:, 1:,:]
        )
        return output
