import torch
from typing import Tuple, List, Optional, Sequence

from pythae.models.base.base_model import ModelOutput
from pythae.data.datasets import BaseDataset
from torch import nn, Tensor
from pythae.models.vae import VAEConfig
from langvae.arch.vae import LangVAE
from langvae.encoders import AnnotatedSentenceEncoder
from langvae.decoders import SentenceDecoder


class LangCVAE(LangVAE):
    def __init__(
            self,
            model_config: VAEConfig,
            encoder: Optional[AnnotatedSentenceEncoder],
            decoder: Optional[SentenceDecoder]
    ):
        decoder = type(decoder)(decoder.decoder.config.name_or_path,
                                encoder.encoder.config.hidden_size * (encoder.num_annotations + 1),
                                decoder.max_len,
                                decoder.device,
                                decoder.load_in_4bit,
                                decoder.device_map,
                                decoder.max_look_behind)
        decoder.debug = True
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.z2emb = nn.Linear(model_config.latent_dim, encoder.encoder.config.hidden_size, device=self.device)

    def forward(self, inputs: BaseDataset, **kwargs):
        x = inputs["data"]

        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        cond_vars = encoder_output["embedding_lbls"]
        recon_x = self.decoder(torch.cat([torch.cat((self.z2emb(z), cvar), dim=-1) for cvar in cond_vars], dim=-1))["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def encode_z(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        mu, log_var = encoded["embedding"], encoded["log_covariance"]
        cond_vars = encoded["embedding_lbls"]
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z, cond_vars

    def decode_sentences(self, z: Tensor, cond_vars: Sequence[Tensor]) -> List[str]:
        cond_z = torch.cat([torch.cat((self.z2emb(z), cvar), dim=-1) for cvar in cond_vars], dim=-1)
        generated = self.decoder(cond_z)["reconstruction"]
        sents = self.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1), skip_special_tokens=True)

        return sents

