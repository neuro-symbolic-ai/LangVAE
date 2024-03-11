import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from pythae.trainers import BaseTrainerConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor
from pythae.models.vae import VAE, VAEConfig
from pythae.trainers.training_callbacks import TrainingCallback


@torch.jit.script
def vae_nll_loss(recon_x: Tensor,
                 x: Tensor,
                 mu: Tensor,
                 log_var: Tensor,
                 z: Tensor,
                 pad_token_id: int,
                 beta: float,
                 target_kl: float) -> Tuple[Tensor, Tensor, Tensor]:
    x = torch.squeeze(x).to(recon_x.device)
    x_tok_ids = torch.argmax(x, dim=-1)
    mask = (x_tok_ids != pad_token_id).to(torch.int8)

    recon_loss = (F.nll_loss(torch.log(recon_x).view(recon_x.shape[0] * recon_x.shape[1], recon_x.shape[2]),
                             x_tok_ids.view(recon_x.shape[0] * recon_x.shape[1]),
                             reduction="none").sum(dim=-1) * mask).sum(dim=-1) / x.shape[0]

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    kl_mask = (KLD > target_kl).float()
    KLD = beta * (kl_mask * KLD)

    return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


class LangVAE(VAE):
    def __init__(
            self,
            model_config: VAEConfig,
            encoder: Optional[BaseEncoder],
            decoder: Optional[BaseDecoder]
    ):
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.cur_beta: float = 0.0
        self.target_kl = 1.0

    def loss_function(self, recon_x, x, mu, log_var, z):
        losses = vae_nll_loss(recon_x, x, mu, log_var, z, self.decoder.tokenizer.pad_token_id, self.cur_beta, self.target_kl)
        print("\n", [l.item() for l in losses])
        return losses

    def encode_z(self, x: Tensor):
        encoded = self.encoder(x)
        mu, log_var = encoded["embedding"], encoded["log_covariance"]
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z

    def decode_sentences(self, z: Tensor):
        generated = self.decoder(z)["reconstruction"]
        sents = self.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1), skip_special_tokens=True)

        return sents
