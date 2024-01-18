import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class SentenceDecoder(BaseDecoder):
    def __init__(self, model_path: str,
                 latent_size: int,
                 max_len: int,
                 device: str = "cpu",
                 args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)
        self.decoder = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_len = max_len
        embedding_dim = self.decoder.get_input_embeddings().embedding_dim
        # self.context_hidden = nn.Sequential(
        #     nn.Linear(
        #         latent_size,
        #         max_len * embedding_dim * self.decoder.config.n_layer * 2,
        #         device=device
        #     ),
        #     nn.Dropout(p=0.4)
        # )
        self.context_embedder = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2, device=device),
            nn.Dropout(p=0.4),
            nn.Linear(latent_size * 2, max_len * embedding_dim, device=device),
        )
        self.context_attention = nn.Sequential(
            nn.Linear(latent_size, latent_size * 2, device=device),
            nn.Dropout(p=0.4),
            nn.Linear(latent_size * 2, max_len, device=device)
            # nn.Linear(latent_size ** 2, max_len * 2, device=device)

        )

        self.decoder.eval()
        self.dbg_counter = 0

    def forward(self, z: Tensor) -> ModelOutput:
        # Fix for MPS bug
        self.decoder = self.decoder.to(z.device)
        self.context_embedder = self.context_embedder.to(z.device)
        self.context_attention = self.context_attention.to(z.device)

        embeds = self.context_embedder(z).view(z.shape[0], self.max_len, self.decoder.get_input_embeddings().embedding_dim)
        # past = [
        #     tuple([h.view(-1,
        #                   self.decoder.config.n_head,
        #                   self.max_len,
        #                   self.decoder.get_input_embeddings().embedding_dim // self.decoder.config.n_head)
        #            for h in v.chunk(2, dim=-1)])
        #     for v in self.context_hidden(z).chunk(self.decoder.config.n_layer, dim=-1)
        # ]
        context_attn = torch.round(F.sigmoid(self.context_attention(z)))
        decoded = self.decoder(
            inputs_embeds=embeds,
            # past_key_values=tuple(past),
            attention_mask=context_attn
        )
        generated = F.softmax(decoded.logits, dim=-1)

        # Debug print (outputs)
        # if (self.dbg_counter % 100 == 0):
        #     dec_text = [" ".join(self.tokenizer.convert_ids_to_tokens(batch)) for batch in torch.argmax(generated, dim=-1)]
        #     print("\n".join(dec_text[:2]))
        # self.dbg_counter += 1

        output = ModelOutput(
            reconstruction=generated
        )
        return output
