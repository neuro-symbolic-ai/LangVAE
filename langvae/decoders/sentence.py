import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class SentenceDecoder(BaseDecoder):
    def __init__(self, model_path: str,
                 latent_size: int,
                 max_len: int,
                 device: str = "cpu",
                 load_in_4bit: bool = False,
                 device_map: str = None,
                 max_look_behind: int = 20,
                 args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)
        if (device.startswith("cuda") and device_map):
            self.decoder = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=load_in_4bit,
                                                                torch_dtype="auto", device_map=device_map)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=load_in_4bit, torch_dtype="auto")
            if (not load_in_4bit):
                self.decoder = self.decoder.to(device)

        self.load_in_4bit = load_in_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_len = max_len
        self.max_look_behind = max_look_behind
        self.device = self.decoder.device
        embedding_dim = self.decoder.get_input_embeddings().embedding_dim

        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.pad_token_id] * 2, dtype=torch.int64, device=self.device), dim=-1)
        pkv = self.decoder(dec_ids).past_key_values
        self.pkv_dims = pkv[0][0].shape[1:]
        self.pkv_dtype = pkv[0][0].dtype

        self.context_embedder = nn.Linear(latent_size, max_len * embedding_dim, dtype=self.pkv_dtype, device=self.device)
        self.context_hidden = nn.Linear(
            latent_size,
            self.decoder.config.hidden_size * self.decoder.config.num_hidden_layers * 2, # self.pkv_dims[0] * self.pkv_dims[1] * self.pkv_dims[2] == self.decoder.config.hidden_size
            dtype=self.pkv_dtype,
            device=self.device
        )
        self.dropout = nn.Dropout(p=0.2)

        self.decoder.eval()
        self.dbg_counter = 0
        self.debug = False

    def forward(self, z: Tensor) -> ModelOutput:
        # Fix for pythae device allocation bug
        if (not self.load_in_4bit):
            self.decoder = self.decoder.to(self.device)
        else:
            self.device = self.decoder.device
        self.context_hidden = self.context_hidden.to(self.device)
        self.context_embedder = self.context_embedder.to(self.device)
        z = z.to(self.pkv_dtype).to(self.device)

        embedding_dim = self.decoder.get_input_embeddings().embedding_dim
        context_embeds = self.dropout(self.context_embedder(z)).view(-1, self.max_len, embedding_dim)
        past = [
            tuple([h.view(-1,
                          self.pkv_dims[0],
                          self.pkv_dims[1],
                          self.pkv_dims[2])
                   for h in v.chunk(2, dim=-1)])
            for v in self.dropout(self.context_hidden(z)).chunk(self.decoder.config.num_hidden_layers, dim=-1)
        ]

        generated = torch.zeros(z.shape[0], self.max_len + 1, self.decoder.config.vocab_size, device=self.device)
        dec_ids = torch.unsqueeze(torch.tensor([self.tokenizer.pad_token_id] * z.shape[0], dtype=torch.int64, device=self.device), dim=-1)
        decoded = self.decoder(input_ids=dec_ids, past_key_values=tuple(past))
        generated[:, 0,:] += F.one_hot(dec_ids, num_classes=generated.shape[-1]).squeeze()
        past_dec = [None] * self.decoder.config.num_hidden_layers

        for i in range(self.max_len):
            for layer_idx in range(self.decoder.config.num_hidden_layers):
                past_dec[layer_idx] = (
                    past[layer_idx][0] + decoded.past_key_values[layer_idx][0][:, :, 1:, :],
                    past[layer_idx][1] + decoded.past_key_values[layer_idx][1][:, :, 1:, :]
                )

            # decoded = self.decoder(input_ids=generated[:, max(0, i-self.max_look_behind):i+1, :].argmax(dim=-1),
            #                        past_key_values=tuple(past_dec))
            gen_ids = generated[:, max(0, i-1):i+1, :].argmax(dim=-1)
            embeds = self.decoder.get_input_embeddings()(gen_ids) + context_embeds[:, max(0, i-1):i+1, :]
            decoded = self.decoder(inputs_embeds=embeds, past_key_values=tuple(past_dec))
            generated[:, i+1,:] = F.softmax(decoded.logits[:, -1, :], dim=-1)


        # Debug print (outputs)
        if (self.debug):
            if (self.dbg_counter % 100 == 0):
                print("\n".join([s.replace(self.tokenizer.pad_token, "|#|") for s in self.tokenizer.batch_decode(torch.argmax(generated, dim=-1))]))
            self.dbg_counter += 1

        output = ModelOutput(
            reconstruction=generated[:, 1:,:]
        )
        return output