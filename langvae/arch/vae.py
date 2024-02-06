import torch
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from pythae.models.vae import VAE


@torch.jit.script
def vae_nll_loss(recon_x: Tensor,
                 x: Tensor,
                 mu: Tensor,
                 log_var: Tensor,
                 z: Tensor,
                 pad_token_id: int,
                 target_kl: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
    x = torch.squeeze(x).to(recon_x.device)
    x_tok_ids = torch.argmax(x, dim=-1)
    mask = (x_tok_ids != pad_token_id).to(torch.int8)

    recon_loss = (F.nll_loss(torch.log(recon_x).view(recon_x.shape[0] * recon_x.shape[1], recon_x.shape[2]),
                             x_tok_ids.view(recon_x.shape[0] * recon_x.shape[1]),
                             reduction="none").sum(dim=-1) * mask).sum(dim=-1) / x.shape[0]

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    kl_mask = (KLD > target_kl).float()
    KLD = kl_mask * KLD

    return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


class LangVAE(VAE):
    def loss_function(self, recon_x, x, mu, log_var, z):
        return vae_nll_loss(recon_x, x, mu, log_var, z, self.decoder.tokenizer.pad_token_id)